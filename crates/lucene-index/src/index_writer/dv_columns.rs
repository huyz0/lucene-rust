//! A flush's doc-values columns in one pass over its documents.
//!
//! [`super::IndexWriter::collect_dense_column`] builds one column by scanning every
//! document's doc values for that field, so a flush of `F` doc-values
//! fields read every document's values `F` times, and it copied each
//! SORTED_SET value and gave each SORTED_NUMERIC and SORTED_SET document a
//! vector of its own. This routes each value to its field's column as the
//! documents are read once, keeps SORTED_NUMERIC values in one flat array,
//! and copies each distinct SORTED_SET value once, into the dictionary --
//! the shapes `SortedNumericDocValuesWriter` and `SortedSetDocValuesWriter`
//! buffer. The columns are the ones a merge writes
//! ([`SortedNumericColumn`], [`SortedSetOrdsColumn`]),
//! so the bytes written are the ones the per-field columns gave.
//!
//! BINARY and SORTED columns, and any column whose documents carry a value
//! of the wrong kind, are left to [`super::IndexWriter::collect_dense_column`],
//! which reports the error exactly as before.

use lucene_codecs::doc_values::{SortedNumericColumn, SortedSetOrdsColumn};
use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::stored_fields::FieldValue;
use lucene_util::fx_hash::FxHashMap;

use super::{DenseColumn, DocValueFields, DocValuesFieldConfig};

/// Field numbers up to this are routed by index; a schema with a larger one
/// takes the per-field path.
const MAX_ROUTED_FIELD: usize = 1 << 16;
const NO_SLOT: u32 = u32::MAX;

enum Builder<'d> {
    /// Left to the per-field path.
    Skip,
    /// The first value of each document, as `collect_numeric_values` keeps.
    Numeric {
        pairs: Vec<(i32, i64)>,
    },
    SortedNumeric(SortedNumericColumn),
    SortedSet {
        docs: Vec<i32>,
        counts: Vec<u32>,
        values: Vec<&'d [u8]>,
    },
}

/// The last document a column took a value from, `-1` before any.
struct Slot<'d> {
    builder: Builder<'d>,
    last: i32,
}

/// Every column of `configs` this pass builds, in `configs`' order; `None`
/// for one it leaves to the per-field path, or for all of them when a value
/// has the wrong kind, a document number overflows, or a field number is
/// too large to route.
pub(super) fn collect_columns<D: DocValueFields>(
    docs: &[D],
    configs: &[DocValuesFieldConfig],
) -> Option<Vec<Option<DenseColumn>>> {
    // An empty flush keeps the per-field path's bytes.
    if docs.is_empty() {
        return None;
    }
    let max_field = configs.iter().map(|c| c.field_number).max()?;
    let width = usize::try_from(max_field).ok()?.checked_add(1)?;
    if width > MAX_ROUTED_FIELD || configs.iter().any(|c| c.field_number < 0) {
        return None;
    }
    let mut slot_of = vec![NO_SLOT; width];
    let mut slots: Vec<Slot<'_>> = Vec::with_capacity(configs.len());
    for (i, c) in configs.iter().enumerate() {
        let n = usize::try_from(c.field_number).ok()?;
        // Each per-field column reads its own number's values, so two
        // configs of one number cannot share a route: leave them all to
        // that path (which refuses them).
        let slot = slot_of.get_mut(n)?;
        if *slot != NO_SLOT {
            return None;
        }
        *slot = u32::try_from(i).ok()?;
        let builder = match c.doc_values_type {
            DocValuesType::Numeric => Builder::Numeric {
                pairs: Vec::with_capacity(docs.len()),
            },
            DocValuesType::SortedNumeric => Builder::SortedNumeric(SortedNumericColumn {
                docs: Vec::with_capacity(docs.len()),
                counts: Vec::with_capacity(docs.len()),
                values: Vec::with_capacity(docs.len()),
            }),
            DocValuesType::SortedSet => Builder::SortedSet {
                docs: Vec::with_capacity(docs.len()),
                counts: Vec::with_capacity(docs.len()),
                values: Vec::with_capacity(docs.len()),
            },
            _ => Builder::Skip,
        };
        slots.push(Slot { builder, last: -1 });
    }
    for (d, doc) in docs.iter().enumerate() {
        let d = i32::try_from(d).ok()?;
        for f in doc.dv_fields() {
            let Some(&s) = usize::try_from(f.field_number)
                .ok()
                .and_then(|n| slot_of.get(n))
            else {
                continue;
            };
            let Some(slot) = slots.get_mut(s as usize) else {
                continue;
            };
            let first = slot.last != d;
            match &mut slot.builder {
                Builder::Skip => {}
                Builder::Numeric { pairs } => {
                    if first {
                        pairs.push((d, numeric(&f.value)?));
                    }
                }
                Builder::SortedNumeric(c) => {
                    let v = numeric(&f.value)?;
                    if first {
                        c.docs.push(d);
                        c.counts.push(0);
                    }
                    c.values.push(v);
                    let k = c.counts.last_mut()?;
                    *k = k.saturating_add(1);
                }
                Builder::SortedSet {
                    docs: set_docs,
                    counts,
                    values,
                } => {
                    let v = match &f.value {
                        FieldValue::String(s) => s.as_bytes(),
                        FieldValue::Binary(b) => b.as_slice(),
                        _ => return None,
                    };
                    if first {
                        set_docs.push(d);
                        counts.push(0);
                    }
                    values.push(v);
                    let k = counts.last_mut()?;
                    *k = k.saturating_add(1);
                }
            }
            slot.last = d;
        }
    }
    let dense = docs.len();
    Some(
        configs
            .iter()
            .zip(slots)
            .map(|(c, slot)| finish(c.field_number, slot.builder, dense))
            .collect(),
    )
}

fn numeric(v: &FieldValue) -> Option<i64> {
    match v {
        FieldValue::Int(v) => Some(i64::from(*v)),
        FieldValue::Long(v) => Some(*v),
        _ => None,
    }
}

fn finish(n: i32, builder: Builder<'_>, max_doc: usize) -> Option<DenseColumn> {
    Some(match builder {
        Builder::Skip => return None,
        Builder::Numeric { pairs } => {
            if pairs.len() == max_doc {
                DenseColumn::Numeric(n, pairs.into_iter().map(|(_, v)| v).collect())
            } else {
                DenseColumn::SparseNumeric(n, pairs)
            }
        }
        Builder::SortedNumeric(mut c) => {
            // `SortedNumericDocValuesWriter.finishCurrentDoc`: each
            // document's values ascending.
            let mut at = 0usize;
            for &k in &c.counts {
                let end = at.saturating_add(k as usize);
                if let Some(run) = c.values.get_mut(at..end) {
                    run.sort_unstable();
                }
                at = end;
            }
            DenseColumn::SortedNumericColumn(n, c)
        }
        Builder::SortedSet {
            docs,
            counts,
            values,
        } => DenseColumn::SortedSetOrds(n, sorted_set(docs, counts, &values)),
    })
}

/// `SortedSetDocValuesWriter`: the distinct values hashed in first, only
/// those sorted and copied, each document's ordinals ascending and distinct.
fn sorted_set(docs: Vec<i32>, mut counts: Vec<u32>, values: &[&[u8]]) -> SortedSetOrdsColumn {
    let mut ids: FxHashMap<&[u8], u32> = FxHashMap::default();
    let mut distinct: Vec<&[u8]> = Vec::new();
    let value_ids: Vec<u32> = values
        .iter()
        .map(|&v| {
            *ids.entry(v).or_insert_with(|| {
                distinct.push(v);
                u32::try_from(distinct.len().saturating_sub(1)).unwrap_or(u32::MAX)
            })
        })
        .collect();
    let mut order: Vec<u32> = (0..u32::try_from(distinct.len()).unwrap_or(u32::MAX)).collect();
    order.sort_unstable_by(|&a, &b| distinct[a as usize].cmp(distinct[b as usize]));
    let mut ord_of = vec![0i64; distinct.len()];
    for (ord, &id) in order.iter().enumerate() {
        ord_of[id as usize] = ord as i64;
    }
    let dict = order
        .iter()
        .map(|&id| distinct[id as usize].to_vec())
        .collect();
    let mut ords: Vec<i64> = Vec::with_capacity(value_ids.len());
    let mut run: Vec<i64> = Vec::new();
    let mut at = 0usize;
    for count in counts.iter_mut() {
        let end = at.saturating_add(*count as usize);
        run.clear();
        run.extend(
            value_ids
                .get(at..end)
                .unwrap_or_default()
                .iter()
                .map(|&id| ord_of[id as usize]),
        );
        run.sort_unstable();
        run.dedup();
        ords.extend_from_slice(&run);
        *count = u32::try_from(run.len()).unwrap_or(u32::MAX);
        at = end;
    }
    SortedSetOrdsColumn {
        dict,
        docs,
        counts,
        ords,
    }
}

#[cfg(test)]
mod tests {
    use lucene_codecs::doc_values::{self, Lucene90DocValuesFormat};
    use lucene_codecs::field_infos::DocValuesType;
    use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};

    use super::super::{DenseColumn, DocValuesFieldConfig, IndexWriter};

    fn config(n: i32, t: DocValuesType, skip: bool) -> DocValuesFieldConfig {
        DocValuesFieldConfig {
            name: format!("f{n}"),
            field_number: n,
            doc_values_type: t,
            skip_index: skip,
        }
    }

    fn v(n: i32, value: FieldValue) -> StoredField {
        StoredField {
            field_number: n,
            value,
        }
    }

    /// What the per-field path writes for the same documents.
    fn per_field(
        docs: &[Document],
        configs: &[DocValuesFieldConfig],
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let columns: Vec<DenseColumn> = configs
            .iter()
            .map(|c| IndexWriter::collect_dense_column(docs, c).unwrap())
            .collect();
        let fields: Vec<_> = columns.iter().map(DenseColumn::as_dense_field).collect();
        let skips: Vec<i32> = configs
            .iter()
            .filter(|c| c.skip_index)
            .map(|c| c.field_number)
            .collect();
        let format = Lucene90DocValuesFormat::default();
        doc_values::write_fields_with_format(
            &fields,
            &skips,
            docs.len() as i32,
            &[7u8; 16],
            &format!("{}_0", format.name()),
            format,
        )
        .unwrap()
    }

    #[test]
    fn one_pass_columns_write_the_per_field_bytes() {
        use DocValuesType::*;
        let mut docs = Vec::new();
        for d in 0..300i64 {
            let mut fields = vec![v(5, FieldValue::Long(d * 3 % 17))];
            if d % 3 != 0 {
                // Sparse sorted-numeric, values out of order, repeated.
                fields.push(v(2, FieldValue::Long(100 - d)));
                fields.push(v(2, FieldValue::Int(-(d as i32))));
                fields.push(v(2, FieldValue::Long(100 - d)));
            }
            // Dense sorted-numeric, one value.
            fields.push(v(3, FieldValue::Long(d << 20)));
            // Dense sorted-set: repeats within a document, strings and bytes.
            fields.push(v(4, FieldValue::Binary(format!("k{}", d % 7).into_bytes())));
            fields.push(v(4, FieldValue::String(format!("k{}", d % 7))));
            if d % 5 == 0 {
                fields.push(v(4, FieldValue::String(format!("z{}", d % 11))));
                // Sparse numeric, and a second value the column ignores.
                fields.push(v(6, FieldValue::Long(d)));
                fields.push(v(6, FieldValue::Long(-1)));
                // Sparse single-valued sorted-set.
                fields.push(v(7, FieldValue::Binary(vec![(d % 4) as u8])));
            }
            // Binary, left to the per-field path.
            fields.push(v(8, FieldValue::Binary(d.to_le_bytes().to_vec())));
            // A field no column asks for.
            fields.push(v(1, FieldValue::Long(d)));
            fields.reverse();
            docs.push(Document { fields });
        }
        let configs = [
            config(5, Numeric, false),
            config(2, SortedNumeric, true),
            config(3, SortedNumeric, false),
            config(4, SortedSet, true),
            config(6, Numeric, false),
            config(7, SortedSet, false),
            config(8, Binary, false),
        ];
        let routed = super::collect_columns(&docs, &configs).unwrap();
        assert!(routed[6].is_none(), "binary stays on the per-field path");
        let got = IndexWriter::build_doc_values_output(
            &docs,
            &configs,
            &[7u8; 16],
            0,
            Lucene90DocValuesFormat::default(),
        )
        .unwrap();
        assert_eq!(got, per_field(&docs, &configs));
    }

    #[test]
    fn a_value_of_the_wrong_kind_leaves_every_column_to_the_per_field_path() {
        let docs = vec![Document {
            fields: vec![v(1, FieldValue::Binary(b"x".to_vec()))],
        }];
        let configs = [
            config(0, DocValuesType::SortedSet, false),
            config(1, DocValuesType::SortedNumeric, false),
        ];
        assert!(super::collect_columns(&docs, &configs).is_none());
        let set = vec![Document {
            fields: vec![v(0, FieldValue::Long(1))],
        }];
        assert!(super::collect_columns(&set, &configs).is_none());
        let err = IndexWriter::build_doc_values_output(
            &docs,
            &configs,
            &[7u8; 16],
            0,
            Lucene90DocValuesFormat::default(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn empty_flushes_and_repeated_numbers_take_the_per_field_path() {
        let configs = [config(0, DocValuesType::SortedSet, false)];
        assert!(super::collect_columns::<Document>(&[], &configs).is_none());
        let docs = vec![Document {
            fields: vec![v(2, FieldValue::Long(1))],
        }];
        let twice = [
            config(2, DocValuesType::SortedNumeric, false),
            config(2, DocValuesType::Numeric, false),
        ];
        assert!(super::collect_columns(&docs, &twice).is_none());
    }

    #[test]
    fn field_numbers_too_large_to_route_take_the_per_field_path() {
        let docs = vec![Document {
            fields: vec![v(1 << 20, FieldValue::Long(1))],
        }];
        let configs = [config(1 << 20, DocValuesType::Numeric, false)];
        assert!(super::collect_columns(&docs, &configs).is_none());
    }
}
