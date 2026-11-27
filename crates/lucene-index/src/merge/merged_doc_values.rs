use super::*;

/// One merged doc-values field, resolved into exactly the shape
/// [`lucene_codecs::doc_values::write_dense_fields`] takes -- the merged
/// segment's whole `.dvm`/`.dvd`/`.dvs` is one call over a slice of these,
/// which is what a real multi-field `Lucene90DocValuesFormat` segment is.
pub(super) enum MergedDocValuesField {
    /// Every merged document has a value.
    Numeric(i32, Vec<i64>),
    /// `(merged doc id, value)` for the documents that have one --
    /// `SortField.setMissingValue`'s normal case, and the shape a
    /// multi-tier index sort's tiers routinely take.
    SparseNumeric(i32, Vec<(i32, i64)>),
    /// BINARY and SORTED_NUMERIC, dense or sparse, streamed in merged-doc
    /// order into one flat buffer each.
    Binary(i32, doc_values::BinaryColumn),
    SortedNumeric(i32, doc_values::SortedNumericColumn),
    /// SORTED and SORTED_SET, dense or sparse, with the merged dictionary
    /// already built by [`OrdinalMap`] and every document's ordinals
    /// remapped into it.
    SortedOrds(i32, doc_values::SortedOrdsColumn),
    SortedSetOrds(i32, doc_values::SortedSetOrdsColumn),
}

impl MergedDocValuesField {
    pub(super) fn as_dense_field(&self) -> doc_values::DenseField<'_> {
        match self {
            MergedDocValuesField::Numeric(n, v) => doc_values::DenseField::Numeric(*n, v),
            MergedDocValuesField::SparseNumeric(n, v) => {
                doc_values::DenseField::SparseNumeric(*n, v)
            }
            MergedDocValuesField::Binary(n, c) => doc_values::DenseField::BinaryColumn(*n, c),
            MergedDocValuesField::SortedNumeric(n, c) => {
                doc_values::DenseField::SortedNumericColumn(*n, c)
            }
            MergedDocValuesField::SortedOrds(n, c) => doc_values::DenseField::SortedOrds(*n, c),
            MergedDocValuesField::SortedSetOrds(n, c) => {
                doc_values::DenseField::SortedSetOrds(*n, c)
            }
        }
    }
}

/// Every merged field number some source has data of this kind for, ascending
/// -- the candidate list every `merge_*_doc_values` function starts from.
///
/// A source with no live documents is skipped: it contributes nothing to the
/// merged output, so whatever doc-values fields it happens to carry cannot
/// affect it (the same exemption is applied again when checking whether a
/// source is missing a field the merge needs).
pub(super) fn doc_values_candidates<'a, T: 'a>(
    sources: &'a [MergeSource<'a>],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    per_source_fields: impl Fn(&'a MergeSource<'a>) -> &'a [T],
    field_number: impl Fn(&T) -> i32,
) -> Vec<i32> {
    let mut candidates: Vec<i32> = Vec::new();
    for ((source, map), live_ids) in sources.iter().zip(per_source_maps).zip(per_source_live_ids) {
        if live_ids.is_empty() {
            continue;
        }
        for field in per_source_fields(source) {
            if let Some(&merged_number) = map.get(&field_number(field)) {
                candidates.push(merged_number);
            }
        }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

/// Each source's `fields`, keyed by the source's own field number (the first
/// entry wins, as the linear `find` this replaces did), and empty for a
/// source with no live documents -- built once per merge stage so resolving
/// a candidate field in every source is a hash lookup rather than a scan of
/// the source's whole field list per (field, source) pair.
pub(super) fn index_source_fields<'a, T>(
    sources: &'a [MergeSource<'a>],
    per_source_live_ids: &[Vec<i32>],
    fields: impl Fn(&'a MergeSource<'a>) -> &'a [T],
    field_number: impl Fn(&T) -> i32,
) -> Vec<HashMap<i32, &'a T>> {
    sources
        .iter()
        .zip(per_source_live_ids)
        .map(|(source, live_ids)| {
            let mut by_number = HashMap::new();
            if !live_ids.is_empty() {
                for field in fields(source) {
                    by_number.entry(field_number(field)).or_insert(field);
                }
            }
            by_number
        })
        .collect()
}

/// Merges numeric doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a NUMERIC
/// column for, with each field's values in the same doc order `doc_order`
/// gives -- so an index-sorted merge's columns are ordered by the merged
/// segment's physical order, not by source.
///
/// # Multiple fields, and per-document sparsity
///
/// There is no single-field-per-merge limit: every candidate is merged, and
/// they all land in the one `.dvm`/`.dvd` the caller writes. That is what a
/// **multi-tier index sort** needs (a second tier is a second NUMERIC
/// column), and it is what a real segment looks like.
///
/// A live document with **no** value for the field is not an error either: it
/// is `SortField.setMissingValue`'s normal case, and the merged column is then
/// written sparsely through the same `IndexedDISI` + values body
/// `Lucene90DocValuesConsumer.writeValues` uses; so is a source with no column
/// for the field at all (see the comment in the body). A field for which **no**
/// merged document has a value is still written -- as
/// `Lucene90DocValuesConsumer.writeValues`' all-missing column
/// (`docsWithFieldOffset = -2`), not omitted -- because omitting it would zero
/// the field's `DocValuesType` in the merged `.fnm` and, for an index-sort
/// tier, leave a segment whose `.si` claims an order nothing can re-derive.
pub(super) fn merge_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.numeric_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.numeric_doc_values,
        |f: &SourceNumericDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        // Resolve each contributing source's entry once, up front --
        // `doc_order` may interleave sources in any order (a k-way sorted
        // merge does), so the per-source resolution can no longer be folded
        // into a single linear pass over `per_source_live_ids` the way
        // concatenation could.
        // `DocValuesConsumer.getMergedNumericDocValues`: a source with no
        // NUMERIC column for this field is simply **not a sub** --
        //
        // ```java
        // FieldInfo readerFieldInfo = mergeState.fieldInfos[i].fieldInfo(mergeFieldInfo.name);
        // if (readerFieldInfo != null
        //     && readerFieldInfo.getDocValuesType() == DocValuesType.NUMERIC) {
        //   values = docValuesProducer.getNumeric(readerFieldInfo);
        // }
        // if (values != null) { subs.add(...); }
        // ```
        //
        // -- so every one of its documents comes out of the merge with no
        // value, which is the sparse column `SortField.setMissingValue`
        // already models and `write_dense_fields` already writes. Two ordinary
        // cases depend on it: a field added to the schema after some segments
        // were flushed, and a doc-values **update** against a field whose base
        // flush wrote no column.
        //
        // It is safe not to treat this as a caller error because a better
        // detector for the same class runs one level up:
        // [`check_format_coverage`] refuses a merge whose caller never opened
        // a format a source's own `.si` lists, and the `debug_assert!` in
        // `IndexWriter::execute_merge` pins that every entry an opened `.dvm`
        // declares reaches the merge. A silently-dropped column can no longer
        // hide behind this branch, so the branch can have Java's meaning.
        let per_source_entry: Vec<Option<&SourceNumericDocValues>> =
            per_source_entries(&by_number, reverse_maps, merged_field_number);

        // One `NumericReader` per source, not a `numeric_value` call per
        // document: that free function allocates a fresh `DisiCursor` and
        // re-walks the docs-with-field region from its start every time, and
        // its own doc comment names a *sort* as the caller that must not do
        // that. `doc_order` visits each source's documents in ascending order
        // (`build_doc_id_maps` is monotone within a source), which is exactly
        // the forward-only access the cursor wants.
        let mut per_source_reader: Vec<Option<doc_values::NumericReader>> = per_source_entry
            .iter()
            .map(|entry| entry.map(|e| doc_values::NumericReader::new(e.data, &e.entry)))
            .collect();
        // Dense until the first document without a value; that document
        // converts what is there once (every earlier document has a value,
        // so `dense[d]` is merged doc `d`'s) and the rest goes sparse only.
        let mut dense: Vec<i64> = Vec::with_capacity(doc_order.len());
        let mut sparse: Option<Vec<(i32, i64)>> = None;
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            // No reader for this source means the source has no column for
            // this field at all: every one of its documents is missing a
            // value, exactly like a document the column itself skips.
            let value = match per_source_reader[src_idx].as_mut() {
                Some(reader) => reader.value(doc_id)?,
                None => None,
            };
            let pruned = sources[src_idx]
                .pruned_field
                .zip(per_source_entry[src_idx])
                .is_some_and(|(pruned, entry)| pruned.drops(entry.entry.field_number, doc_id));
            let value = value.filter(|_| !pruned).map(|v| {
                match (sources[src_idx].numeric_floor, per_source_entry[src_idx]) {
                    (Some(floor), Some(entry)) => floor.apply(entry.entry.field_number, doc_id, v),
                    _ => v,
                }
            });
            match (value, sparse.as_mut()) {
                (Some(value), None) => dense.push(value),
                (Some(value), Some(sparse)) => sparse.push((merged_doc_id as i32, value)),
                (None, None) => {
                    sparse = Some(
                        std::mem::take(&mut dense)
                            .into_iter()
                            .enumerate()
                            .map(|(doc, value)| (doc as i32, value))
                            .collect(),
                    );
                }
                (None, Some(_)) => {}
            }
        }
        if let Some(sparse) = sparse {
            // Including when `sparse` is empty. `Lucene90DocValuesConsumer`
            // is called for every field the merged `FieldInfos` gives a
            // `DocValuesType`, and `writeValues` records an all-missing
            // column as `docsWithFieldOffset = -2` -- it does not omit the
            // field. Omitting it here would zero the field's
            // `DocValuesType` in the merged `.fnm`
            // (`describe_written_files`), which for an **index-sort tier** is
            // a segment whose `.si` claims an order no reader can re-derive:
            // real Lucene's `DocValues.getNumeric` throws for a field whose
            // `FieldInfo` exists but declares no doc values, so
            // `CheckIndex.testSort` fails rather than degrading, and this
            // port's own `execute_merge` could never merge the segment again
            // (`Error::MergeSortColumnMissing`, propagated out of `commit`).
            merged.push(MergedDocValuesField::SparseNumeric(
                merged_field_number,
                sparse,
            ));
        } else {
            merged.push(MergedDocValuesField::Numeric(merged_field_number, dense));
        }
    }
    Ok(merged)
}

/// How many doc-values columns a merge builds at once.
pub(super) const DOC_VALUES_MERGE_WINDOW: usize = 4;

/// Every merged field number some source has doc-values data of any kind for,
/// ascending: what the five `merge_*_doc_values` stages would produce between
/// them.
pub(super) fn merged_doc_values_field_numbers(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
) -> Vec<i32> {
    let mut numbers = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.numeric_doc_values,
        |f| f.entry.field_number,
    );
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.binary_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_numeric_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.extend(doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_set_doc_values,
        |f| f.entry.field_number,
    ));
    numbers.sort_unstable();
    numbers
}

/// Each contributing source's entry for `merged_field_number`, or `None` for
/// a source with no live documents or no such column --
/// `DocValuesConsumer.getMerged*DocValues`' rule that a source whose
/// `FieldInfo` lacks the column is simply **not a sub**, so every one of its
/// documents comes out of the merge with no value (see
/// [`merge_numeric_doc_values`] for why that is safe to rely on here).
///
/// `by_number` is [`index_source_fields`]' per-source index and
/// `reverse_maps` [`invert_field_number_maps`]' merged -> original maps, so
/// this is two hash lookups per source.
pub(super) fn per_source_entries<'a, T>(
    by_number: &[HashMap<i32, &'a T>],
    reverse_maps: &[HashMap<i32, i32>],
    merged_field_number: i32,
) -> Vec<Option<&'a T>> {
    by_number
        .iter()
        .zip(reverse_maps)
        .map(|(fields, reverse)| {
            let original = reverse.get(&merged_field_number)?;
            fields.get(original).copied()
        })
        .collect()
}

/// Merges BINARY doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a BINARY
/// column for, in `doc_order` -- `DocValuesConsumer.mergeBinaryField`. A
/// document with no value (its source has no column, or the column skips
/// it) is missing from the merged column, which is then written sparsely.
pub(super) fn merge_binary_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.binary_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.binary_doc_values,
        |f: &SourceBinaryDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        // One forward cursor per source (`doc_order` visits each source's
        // documents in ascending order), as `getMergedBinaryDocValues`
        // iterates each sub.
        let mut readers: Vec<Option<doc_values::BinaryReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| doc_values::BinaryReader::new(e.data, &e.entry)))
            .collect();
        let mut column = doc_values::BinaryColumn::default();
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            if let Some(value) = reader.value_ref(doc_id)? {
                column.docs.push(merged_doc_id as i32);
                column.lengths.push(value.len() as u32);
                column.bytes.extend_from_slice(value);
            }
        }
        merged.push(MergedDocValuesField::Binary(merged_field_number, column));
    }
    Ok(merged)
}

/// `OrdinalMap`: the merged dictionary of several sources' sorted, unique
/// dictionaries, and each source's ordinal -> merged-ordinal map.
///
/// Java's walks the sources' `TermsEnum`s through a priority queue ordered by
/// current term, handing out the next global ordinal each time the smallest
/// term changes; this is that walk, over dictionaries already decoded. A
/// source with no column (`None`) contributes nothing and gets an empty map.
pub(super) struct OrdinalMap {
    dict: Vec<Vec<u8>>,
    /// `segment_to_global[source][ord]`; `None` for a term no live document
    /// of the source uses.
    segment_to_global: Vec<Vec<Option<i64>>>,
}

impl OrdinalMap {
    /// `OrdinalMap.build` over `source_dicts`, each restricted to its
    /// `live_ords` when it has one -- `mergeSortedField`/`mergeSortedSetField`
    /// hand a source with deletions over as a `BitsFilteredTermsEnum` of the
    /// ordinals its live documents use, so a term only deleted documents
    /// carried leaves the merged dictionary (Java's `CheckIndex` refuses a
    /// dictionary with a term no document uses).
    fn build(source_dicts: Vec<Option<Vec<Vec<u8>>>>, live_ords: Vec<Option<FixedBitSet>>) -> Self {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut segment_to_global: Vec<Vec<Option<i64>>> = source_dicts
            .iter()
            .map(|d| Vec::with_capacity(d.as_ref().map_or(0, Vec::len)))
            .collect();
        let dicts: Vec<Vec<Vec<u8>>> = source_dicts
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect();
        // (current term, source, ord within source), smallest term first.
        let mut queue: BinaryHeap<Reverse<(&[u8], usize, usize)>> = dicts
            .iter()
            .enumerate()
            .filter(|(_, d)| !d.is_empty())
            .map(|(src, d)| Reverse((d[0].as_slice(), src, 0)))
            .collect();
        let mut dict: Vec<Vec<u8>> = Vec::new();
        while let Some(Reverse((term, src, ord))) = queue.pop() {
            let live = live_ords
                .get(src)
                .and_then(Option::as_ref)
                .is_none_or(|live| live.get_doc(i32::try_from(ord).unwrap_or(i32::MAX)));
            if !live {
                segment_to_global[src].push(None);
            } else {
                if dict.last().map(Vec::as_slice) != Some(term) {
                    dict.push(term.to_vec());
                }
                // ARITH: `dict` is non-empty after the push above.
                #[allow(clippy::arithmetic_side_effects)]
                segment_to_global[src].push(Some((dict.len() - 1) as i64));
            }
            // ARITH: `ord` indexes `dicts[src]`, so `ord + 1 <= len`.
            #[allow(clippy::arithmetic_side_effects)]
            let next = ord + 1;
            if let Some(t) = dicts[src].get(next) {
                queue.push(Reverse((t.as_slice(), src, next)));
            }
        }
        Self {
            dict,
            segment_to_global,
        }
    }

    /// `getGlobalOrds(source).get(ord)`; an ordinal outside the source's
    /// dictionary is a corrupt source.
    fn global(&self, source: usize, ord: i64) -> Option<i64> {
        usize::try_from(ord)
            .ok()
            .and_then(|o| self.segment_to_global[source].get(o).copied())
            .flatten()
    }
}

/// The ordinals `source`'s live documents use, for [`OrdinalMap::build`] --
/// `None` when the source has no deletions, as Java hands such a source's
/// terms over unfiltered. `ords` writes one document's ordinals.
pub(super) fn live_ords(
    source: &MergeSource,
    live_ids: &[i32],
    value_count: usize,
    mut ords: impl FnMut(i32, &mut Vec<i64>) -> Result<()>,
) -> Result<Option<FixedBitSet>> {
    if source.live_docs.is_none() {
        return Ok(None);
    }
    let mut used = FixedBitSet::new(value_count);
    let mut scratch = Vec::new();
    for &doc in live_ids {
        scratch.clear();
        ords(doc, &mut scratch)?;
        for &ord in &scratch {
            // An ordinal outside the dictionary stays unset here and is
            // reported by the merge's own lookup.
            if let Ok(ord) = usize::try_from(ord) {
                if ord < value_count {
                    // FBS: `ord < value_count`, the bitset's length.
                    used.set(ord);
                }
            }
        }
    }
    Ok(Some(used))
}

/// Merges SORTED doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a SORTED
/// column for, in `doc_order` -- `DocValuesConsumer.mergeSortedField`: an
/// [`OrdinalMap`] over the sources' dictionaries, then each document's
/// ordinal remapped through it. No term bytes are copied per document and no
/// dictionary is rebuilt by sorting. Dense or sparse as
/// [`merge_binary_doc_values`].
pub(super) fn merge_sorted_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.sorted_doc_values,
        |f: &SourceSortedDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        let dicts: Vec<Option<Vec<Vec<u8>>>> = per_source_entry
            .iter()
            .map(|e| {
                e.map(|e| terms_dict::decode_all_terms(e.data, &e.entry.terms))
                    .transpose()
            })
            .collect::<std::result::Result<_, _>>()?;
        let mut live = Vec::with_capacity(sources.len());
        for (idx, entry) in per_source_entry.iter().enumerate() {
            live.push(match (entry, &dicts[idx]) {
                (Some(e), Some(dict)) => {
                    let mut reader = doc_values::NumericReader::new(e.data, &e.entry.ords);
                    live_ords(
                        &sources[idx],
                        &per_source_live_ids[idx],
                        dict.len(),
                        |doc, out| {
                            out.extend(reader.value(doc)?);
                            Ok(())
                        },
                    )?
                }
                _ => None,
            });
        }
        let map = OrdinalMap::build(dicts, live);
        let mut readers: Vec<Option<doc_values::NumericReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| doc_values::NumericReader::new(e.data, &e.entry.ords)))
            .collect();
        let mut column = doc_values::SortedOrdsColumn::default();
        column.docs.reserve_exact(doc_order.len());
        column.ords.reserve_exact(doc_order.len());
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            let Some(ord) = reader.value(doc_id)? else {
                continue;
            };
            let global =
                map.global(src_idx, ord)
                    .ok_or(Error::SortedDocValuesFieldMissingInSource {
                        merged_field_number,
                    })?;
            column.docs.push(merged_doc_id as i32);
            column.ords.push(global);
        }
        column.dict = map.dict;
        merged.push(MergedDocValuesField::SortedOrds(
            merged_field_number,
            column,
        ));
    }
    Ok(merged)
}

/// Merges SORTED_NUMERIC doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a
/// SORTED_NUMERIC column for, in `doc_order` --
/// `DocValuesConsumer.mergeSortedNumericField`. A document is missing when
/// its source has no column or the column gives it no values.
///
/// A field `remap` names has each document's values rewritten through it.
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_sorted_numeric_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    merged_fields: &[FieldInfo],
    remap: Option<&SortedNumericRemap>,
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_numeric_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.sorted_numeric_doc_values,
        |f: &SourceSortedNumericDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        let mut readers: Vec<Option<doc_values::SortedNumericReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| doc_values::SortedNumericReader::new(e.data, &e.entry)))
            .collect();
        let field_remap = remap.filter(|r| {
            merged_fields
                .iter()
                .find(|f| f.number == merged_field_number)
                .is_some_and(|f| r.fields.contains(&f.name))
        });
        let mut scratch: Vec<i64> = Vec::new();
        let mut column = doc_values::SortedNumericColumn::default();
        column.docs.reserve_exact(doc_order.len());
        column.counts.reserve_exact(doc_order.len());
        column.values.reserve_exact(doc_order.len());
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            reader.values(doc_id, &mut scratch)?;
            if let Some(r) = field_remap {
                r.apply(&mut scratch)?;
            }
            if !scratch.is_empty() {
                column.docs.push(merged_doc_id as i32);
                column.counts.push(scratch.len() as u32);
                column.values.extend_from_slice(&scratch);
            }
        }
        merged.push(MergedDocValuesField::SortedNumeric(
            merged_field_number,
            column,
        ));
    }
    Ok(merged)
}

/// Resolves one live doc's SORTED_SET ordinals for `doc_id`, regardless of
/// whether `entry.kind` collapsed to [`SortedSetKind::Single`] (one ordinal
/// or none) or stayed [`SortedSetKind::Multi`] (zero or more via the same
/// [`SortedNumericEntry`] layout [`doc_values::sorted_numeric_values`]
/// already decodes). Kept for tests and single lookups; a merge holds a
/// [`SortedSetOrdsReader`].
#[cfg(test)]
pub(super) fn sorted_set_doc_ordinals(
    data: &[u8],
    entry: &SortedSetEntry,
    doc_id: i32,
) -> Result<Vec<i64>> {
    match &entry.kind {
        SortedSetKind::Single(sorted) => Ok(doc_values::sorted_ord(data, sorted, doc_id)?
            .into_iter()
            .collect()),
        SortedSetKind::Multi { ords, .. } => {
            Ok(doc_values::sorted_numeric_values(data, ords, doc_id)?)
        }
    }
}

/// A forward cursor over one source's SORTED_SET ordinals, whichever of
/// [`SortedSetKind`]'s two shapes it was written in.
pub(super) enum SortedSetOrdsReader<'a> {
    Single(Box<doc_values::NumericReader<'a>>),
    Multi(Box<doc_values::SortedNumericReader<'a>>),
}

impl<'a> SortedSetOrdsReader<'a> {
    fn new(data: &'a [u8], entry: &'a SortedSetEntry) -> Self {
        match &entry.kind {
            SortedSetKind::Single(sorted) => {
                Self::Single(Box::new(doc_values::NumericReader::new(data, &sorted.ords)))
            }
            SortedSetKind::Multi { ords, .. } => {
                Self::Multi(Box::new(doc_values::SortedNumericReader::new(data, ords)))
            }
        }
    }

    fn ords(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        match self {
            Self::Single(reader) => {
                out.clear();
                out.extend(reader.value(doc)?);
            }
            Self::Multi(reader) => reader.values(doc, out)?,
        }
        Ok(())
    }
}

/// Decodes one source's whole SORTED_SET term dictionary, in ordinal order --
/// the input [`OrdinalMap`] merges.
pub(super) fn sorted_set_source_dict(data: &[u8], entry: &SortedSetEntry) -> Result<Vec<Vec<u8>>> {
    match &entry.kind {
        SortedSetKind::Single(sorted) => Ok(terms_dict::decode_all_terms(data, &sorted.terms)?),
        SortedSetKind::Multi { terms, .. } => Ok(terms_dict::decode_all_terms(data, terms)?),
    }
}

/// Merges SORTED_SET doc-values data across `sources`, one
/// [`MergedDocValuesField`] per merged field number any source has a
/// SORTED_SET column for, in `doc_order` --
/// `DocValuesConsumer.mergeSortedSetField`: an [`OrdinalMap`] over the
/// sources' dictionaries, then each document's ordinals remapped through it.
/// A source's per-document ordinals are ascending and unique, and the map is
/// monotonic, so the remapped ones are too. Dense or sparse as
/// [`merge_binary_doc_values`].
pub(super) fn merge_sorted_set_doc_values(
    sources: &[MergeSource],
    per_source_maps: &[HashMap<i32, i32>],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
    only: Option<i32>,
) -> Result<Vec<MergedDocValuesField>> {
    let candidates = doc_values_candidates(
        sources,
        per_source_maps,
        per_source_live_ids,
        |s| s.sorted_set_doc_values,
        |f| f.entry.field_number,
    );
    let candidates: Vec<i32> = candidates
        .into_iter()
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.sorted_set_doc_values,
        |f: &SourceSortedSetDocValues| f.entry.field_number,
    );
    let mut merged = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        let per_source_entry = per_source_entries(&by_number, reverse_maps, merged_field_number);
        let dicts: Vec<Option<Vec<Vec<u8>>>> = per_source_entry
            .iter()
            .map(|e| {
                e.map(|e| sorted_set_source_dict(e.data, &e.entry))
                    .transpose()
            })
            .collect::<Result<_>>()?;
        let mut live = Vec::with_capacity(sources.len());
        for (idx, entry) in per_source_entry.iter().enumerate() {
            live.push(match (entry, &dicts[idx]) {
                (Some(e), Some(dict)) => {
                    let mut reader = SortedSetOrdsReader::new(e.data, &e.entry);
                    live_ords(
                        &sources[idx],
                        &per_source_live_ids[idx],
                        dict.len(),
                        |doc, out| reader.ords(doc, out),
                    )?
                }
                _ => None,
            });
        }
        let map = OrdinalMap::build(dicts, live);
        let mut readers: Vec<Option<SortedSetOrdsReader>> = per_source_entry
            .iter()
            .map(|e| e.map(|e| SortedSetOrdsReader::new(e.data, &e.entry)))
            .collect();
        let mut scratch: Vec<i64> = Vec::new();
        let mut column = doc_values::SortedSetOrdsColumn::default();
        column.docs.reserve_exact(doc_order.len());
        column.counts.reserve_exact(doc_order.len());
        column.ords.reserve_exact(doc_order.len());
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let Some(reader) = readers[src_idx].as_mut() else {
                continue;
            };
            reader.ords(doc_id, &mut scratch)?;
            if scratch.is_empty() {
                continue;
            }
            for &ord in &scratch {
                column.ords.push(map.global(src_idx, ord).ok_or(
                    Error::SortedSetDocValuesFieldMissingInSource {
                        merged_field_number,
                    },
                )?);
            }
            column.docs.push(merged_doc_id as i32);
            column.counts.push(scratch.len() as u32);
        }
        column.dict = map.dict;
        merged.push(MergedDocValuesField::SortedSetOrds(
            merged_field_number,
            column,
        ));
    }
    Ok(merged)
}

/// Same shape and same rules as [`merge_numeric_doc_values`], for norms --
/// **every** norms field the sources declare, not one.
///
/// Returns one `(merged field number, one value per merged document)` pair
/// per field, in ascending merged-field-number order so a merged `.nvm`'s
/// entry order is a function of the schema rather than of which source
/// happened to be first.
///
/// Every field goes into one `.nvm`/`.nvd` pair through `norms::write_fields`,
/// as `Lucene90NormsConsumer` gets one `addNormsField` call per field.
///
/// Which fields get a column, and what a document with no norm becomes, are
/// both Java's:
///
/// ```java
/// for (FieldInfo mergeFieldInfo : mergeState.mergeFieldInfos) {
///   if (mergeFieldInfo.hasNorms()) {
///     mergeNormsField(mergeFieldInfo, mergeState);
///   }
/// }
/// ```
///
/// -- the candidate list comes from the **merged** field infos
/// (`hasNorms() == !omitNorms && indexOptions != NONE`), not from which
/// sources happen to carry norms files, and `mergeNormsField` then merges
/// only the sources whose own `FieldInfo` says `hasNorms()`:
/// a source that never declared the field, and a document inside a source
/// whose norm column simply has no entry for it, both contribute *nothing*
/// and appear as a gap in the merged column, which is written sparse.
///
/// A segment whose documents do not all carry the field has a sparse column,
/// and merging two segments that disagree about a field is the ordinary case
/// for a schema that grew, so neither is an error.
pub(super) fn merge_norms(
    sources: &[MergeSource],
    merged_fields: &[FieldInfo],
    reverse_maps: &[HashMap<i32, i32>],
    per_source_live_ids: &[Vec<i32>],
    doc_order: &[(usize, i32)],
) -> Result<Vec<MergedNorms>> {
    let candidates: Vec<i32> = merged_fields
        .iter()
        .filter(|f| !f.omit_norms && f.index_options != IndexOptions::None)
        .map(|f| f.number)
        .collect();
    // Same "fully-deleted source can't affect the merged output" exemption
    // as merge_numeric_doc_values: such a source indexes no fields.
    let by_number = index_source_fields(
        sources,
        per_source_live_ids,
        |s| s.norms,
        |f: &SourceNorms| f.entry.field_number,
    );

    let mut merged: Vec<MergedNorms> = Vec::with_capacity(candidates.len());
    for merged_field_number in candidates {
        // Java: NormsConsumer.mergeNormsField retains one NumericDocValues
        // iterator per source. Reuse sparse cursors instead of rescanning each
        // source's document prefix for every merged document. A source whose
        // reverse map has no entry is `readerFieldInfo == null` in
        // `mergeNormsField`: it never declared the field, so it adds no sub.
        let mut per_source_readers: Vec<Option<norms::NormsReader<'_>>> =
            per_source_entries(&by_number, reverse_maps, merged_field_number)
                .into_iter()
                .map(|entry| entry.map(|entry| norms::NormsReader::new(entry.data, &entry.entry)))
                .collect();

        // Dense (missing = 1) for every field, since the postings impacts
        // read it; the sparse wire column only once a document turns out to
        // have no norm, seeded from the all-present prefix.
        let mut values: Vec<i64> = Vec::with_capacity(doc_order.len());
        let mut sparse: Option<Vec<(i32, i64)>> = None;
        for (merged_doc_id, &(src_idx, doc_id)) in doc_order.iter().enumerate() {
            let value = match per_source_readers[src_idx].as_mut() {
                Some(reader) => reader.value(doc_id)?,
                None => None,
            };
            match (value, sparse.as_mut()) {
                (Some(value), None) => values.push(value),
                (Some(value), Some(sparse)) => {
                    values.push(value);
                    sparse.push((merged_doc_id as i32, value));
                }
                (None, sparse_so_far) => {
                    if sparse_so_far.is_none() {
                        sparse = Some(
                            values
                                .iter()
                                .enumerate()
                                .map(|(doc, &value)| (doc as i32, value))
                                .collect(),
                        );
                    }
                    values.push(1);
                }
            }
        }
        merged.push(MergedNorms {
            field_number: merged_field_number,
            values,
            sparse,
        });
    }
    Ok(merged)
}

/// One merged norms field, as [`merge_norms`] returns it.
#[derive(Debug, PartialEq)]
pub(super) struct MergedNorms {
    pub(super) field_number: i32,
    /// One value per merged document, a missing norm as `1` -- what the
    /// postings writer's impacts read, and the `.nvd` column itself when
    /// `sparse` is `None`. A present zero stays zero.
    pub(super) values: Vec<i64>,
    /// `(merged doc id, norm)` for the documents that have one, or `None`
    /// when every document does.
    sparse: Option<Vec<(i32, i64)>>,
}

impl MergedNorms {
    pub(super) fn as_norms_field(&self) -> norms::NormsField<'_> {
        match &self.sparse {
            None => norms::NormsField::Dense(self.field_number, &self.values),
            Some(sparse) => norms::NormsField::Sparse(self.field_number, sparse),
        }
    }

    /// Back to one `Option` per merged document.
    #[cfg(test)]
    pub(super) fn options(&self) -> Vec<Option<i64>> {
        match &self.sparse {
            None => self.values.iter().copied().map(Some).collect(),
            Some(sparse) => {
                let mut out = vec![None; self.values.len()];
                for &(doc, value) in sparse {
                    out[doc as usize] = Some(value);
                }
                out
            }
        }
    }
}
