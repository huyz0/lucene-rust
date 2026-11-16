//! `STUniformSplitTermsWriter.merge` and `STMergingTermsEnum`: one forward
//! scan of each source's shared dictionary, grouped by term and then by the
//! target field number. Document maps remove deletes and rebase documents.
//!
//! Differs: a caller supplies Java's superclass merge as a closure, invoked
//! for index sorting or wrapped/foreign terms. Files are borrowed inputs;
//! this module cannot depend on `lucene-index`. The existing postings reader
//! materializes one source term's IDs/frequencies. Positions, offsets and
//! payloads stream into the merged current term, without an intermediate
//! source-occurrence container. Only that term/field is buffered, and the
//! postings writer keeps its output streams open across terms.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use lucene_util::fixed_bit_set::FixedBitSet;

use super::{StMergingBlockReader, StUniformSplitPostingsFormat, StUniformSplitTermsReader};
use crate::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use crate::postings::{self, DocInput, PayInput, PosInput, PostingsFlags};
use crate::postings_writer::{FieldNorms, FieldPostingsInput, PostingsWriter, TermPostings};
use crate::segment_state::SegmentWriteState;
use crate::uniformsplit::{corrupt, Error, Result, UniformSplitOutput, UniformSplitTermState};

/// One source of `MergeState`: its selected fields, terms and postings, and
/// its old-to-new document map. Slice order is Java's segment index order.
#[derive(Clone, Copy)]
pub struct StMergeSegment<'a> {
    /// The direct shared-terms producer; `None` means wrapped/foreign terms
    /// and selects the caller's superclass merge before reading any input.
    pub reader: Option<&'a StUniformSplitTermsReader>,
    /// `mergeState.fieldInfos[segmentIndex]`, possibly filtered by the
    /// per-field format. Used to find the first eligible shared dictionary.
    pub field_infos: &'a FieldInfos,
    /// `mergeState.docMaps[segmentIndex]`, indexed by the old document ID.
    /// `-1` means deleted; retained IDs must increase across source segments
    /// on the specialized path. Sorted maps belong to the fallback merge.
    pub doc_map: &'a [i32],
    /// The source format instance's `.doc`.
    pub doc: &'a DocInput<'a>,
    /// Its `.pos`, when the source indexes positions.
    pub pos: Option<&'a PosInput<'a>>,
    /// Its `.pay`, when the source indexes offsets or payloads.
    pub pay: Option<&'a PayInput<'a>>,
}

struct SegmentTerms<'a> {
    segment: &'a StMergeSegment<'a>,
    reader: StMergingBlockReader<'a>,
    fields: BTreeMap<i32, &'a FieldInfo>,
    states: Vec<(i32, UniformSplitTermState)>,
}

impl SegmentTerms<'_> {
    /// `SegmentTerms.nextTerm`.
    fn next_term(&mut self) -> Result<Option<Vec<u8>>> {
        let term = self.reader.next()?.map(<[u8]>::to_vec);
        if term.is_some() {
            self.reader.read_field_term_states(&mut self.states)?;
        }
        Ok(term)
    }
}

struct SegmentPostings<'a> {
    source: usize,
    field: &'a FieldInfo,
    state: UniformSplitTermState,
}

/// `STMergingTermsEnum`'s one-term view. Unsupported TermsEnum operations
/// have no Rust entry point; only the sequential postings merge is needed.
struct StMergingTermsEnum;

/// The checked postings walk lends payload slices directly to the target:
/// no `Vec<Position>` per source document or `Vec<u8>` per source payload.
struct MergedOccurrences<'a> {
    term: &'a mut TermPostings,
    offsets: bool,
    payloads: bool,
}

impl postings::OccurrenceSink for MergedOccurrences<'_> {
    const NEEDS_EXTRAS: bool = true;

    fn open_doc(&mut self) {
        self.term.positions.push(Vec::new());
        if self.offsets {
            self.term.offsets.push(Vec::new());
        }
    }

    fn push(&mut self, position: i32, start_offset: i32, end_offset: i32, payload: &[u8]) {
        self.term
            .positions
            .last_mut()
            .expect("open_doc precedes push")
            .push(position);
        if self.offsets {
            self.term
                .offsets
                .last_mut()
                .expect("open_doc precedes push")
                .push((start_offset, end_offset));
        }
        if self.payloads {
            // The walk lends a slice of the decoded u32 payload length;
            // PositionBlock::payload checks its bounds before lending it.
            self.term.payload_lengths.push(payload.len() as u32);
            self.term.payload_bytes.extend_from_slice(payload);
        }
    }
}

impl StMergingTermsEnum {
    /// `MultiSegmentsPostingsEnum.nextDoc`: exhaust each segment in order,
    /// skip docMap == -1, and delegate frequency/position/offset/payload to
    /// that segment's postings. The returned term is ready for writeTerm.
    fn postings(
        term: &[u8],
        postings_sources: &[SegmentPostings<'_>],
        sources: &[SegmentTerms<'_>],
        target: &FieldInfo,
    ) -> Result<TermPostings> {
        let mut merged = TermPostings {
            term: term.to_vec(),
            ..TermPostings::default()
        };
        for source in postings_sources {
            let segment = sources[source.source].segment;
            let docs = sources[source.source].reader.postings(
                source.field.number,
                &source.state,
                segment.doc,
                PostingsFlags::Freqs,
            )?;
            let pos = if target.index_options.subsumes_positions() {
                let pos = segment
                    .pos
                    .ok_or_else(|| corrupt("merge source lacks .pos"))?;
                if (source.field.index_options.subsumes_offsets() || source.field.store_payloads)
                    && segment.pay.is_none()
                {
                    return Err(corrupt("merge source lacks .pay"));
                }
                Some(pos)
            } else {
                None
            };
            let mut wanted = Vec::new();
            for (index, &doc) in docs.docs.iter().enumerate() {
                let mapped = usize::try_from(doc)
                    .ok()
                    .and_then(|doc| segment.doc_map.get(doc))
                    .copied()
                    .ok_or_else(|| corrupt("merge source document is outside its doc map"))?;
                if mapped == -1 {
                    continue;
                }
                if mapped < 0 || merged.docs.last().is_some_and(|&(prev, _)| mapped <= prev) {
                    return Err(corrupt("merge document maps are not increasing"));
                }
                let freq = if target.index_options == IndexOptions::Docs {
                    1
                } else {
                    docs.freqs.get(index).copied().unwrap_or(1)
                };
                merged.docs.push((mapped, freq));
                if pos.is_some() {
                    wanted.push(index);
                }
            }
            if let Some(pos) = pos {
                postings::walk_wanted_occurrences(
                    pos,
                    segment.pay,
                    source.state.meta,
                    &docs.freqs,
                    source.state.total_term_freq,
                    source.field.index_options,
                    source.field.store_payloads,
                    &wanted,
                    &mut MergedOccurrences {
                        term: &mut merged,
                        offsets: target.index_options.subsumes_offsets(),
                        payloads: target.store_payloads,
                    },
                )?;
            }
        }
        Ok(merged)
    }
}

impl StUniformSplitPostingsFormat {
    /// `STUniformSplitTermsWriter.merge(MergeState, NormsProducer)`.
    /// `state.field_infos` are the target merge fields (possibly filtered).
    /// `norms` are already mapped to the target document IDs.
    ///
    /// Java calls its superclass merge for index sorting or terms wrapped
    /// by another producer; `fallback` is that operation supplied by the
    /// owning index layer. It is called once, before opening shared terms.
    ///
    /// # Errors
    /// Source decoding and writing errors; `Corrupted` for invalid document
    /// maps, missing postings streams or inconsistent source metadata;
    /// `IllegalArgument` for a negative target maxDoc; the fallback's errors.
    pub fn merge<F>(
        &self,
        segments: &[StMergeSegment<'_>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
        needs_index_sort: bool,
        fallback: F,
    ) -> Result<UniformSplitOutput>
    where
        F: FnOnce() -> Result<UniformSplitOutput>,
    {
        if needs_index_sort || segments.iter().any(|s| s.reader.is_none()) {
            return fallback();
        }
        let max_doc = usize::try_from(state.max_doc)
            .map_err(|_| Error::IllegalArgument("negative merge maxDoc".into()))?;
        let mut sources = Vec::new();
        for segment in segments {
            let Some(reader) = segment.reader else {
                unreachable!()
            };
            let Some(first) = segment
                .field_infos
                .fields
                .iter()
                .find_map(|field| reader.terms(&field.name))
            else {
                continue;
            };
            let fields = reader
                .field_names()
                .map(|name| {
                    let field = reader.terms(name).expect("reader's own field").field_info();
                    (field.number, field)
                })
                .collect();
            sources.push(SegmentTerms {
                segment,
                reader: first.merging_block_reader(),
                fields,
                states: Vec::new(),
            });
        }
        // `TermIteratorQueue<SegmentTerms>` orders ties by segment index.
        let mut queue = BinaryHeap::new();
        for (index, source) in sources.iter_mut().enumerate() {
            if let Some(term) = source.next_term()? {
                queue.push(Reverse((term, index)));
            }
        }
        let mut fields: Vec<_> = state
            .field_infos
            .fields
            .iter()
            .map(|field| (field.number, 0))
            .collect();
        fields.sort_unstable_by_key(|&(number, _)| number);
        let mut terms = super::StUniformSplitTermsWriter::new(
            state,
            self.base.target_num_block_lines,
            self.base.delta_num_lines,
            self.base.block_encoder.clone(),
            &fields,
        )?;
        let mut postings = PostingsWriter::new(
            state.has_prox(),
            state.has_payloads_or_offsets(),
            state.segment_id,
            state.segment_suffix,
        );
        let mut docs_seen = BTreeMap::<i32, FixedBitSet>::new();
        let mut grouped = Vec::with_capacity(sources.len());
        // Java reuses its per-field source lists. Keep indices rather than
        // borrows of the mutable union walkers so these buffers can live
        // across next_term calls. Only fields active on this term are visited.
        let mut by_field: BTreeMap<i32, Vec<SegmentPostings<'_>>> = fields
            .iter()
            .map(|&(number, _)| (number, Vec::new()))
            .collect();
        let mut active_fields = Vec::with_capacity(fields.len());
        while let Some(Reverse((term, index))) = queue.pop() {
            // `groupByTerm`, then `combineSegmentsFields`.
            grouped.clear();
            grouped.push(index);
            while queue
                .peek()
                .is_some_and(|Reverse((other, _))| *other == term)
            {
                grouped.push(queue.pop().expect("queue is nonempty").0 .1);
            }
            active_fields.clear();
            for &index in &grouped {
                let source = &sources[index];
                for &(number, source_state) in &source.states {
                    let &field = source
                        .fields
                        .get(&number)
                        .ok_or_else(|| corrupt("shared merge line has an unknown field"))?;
                    // `combinePostingsPerField` drops fields removed from
                    // mergeFieldInfos, then orders the rest by target number.
                    if let Some(target) = state.field_infos.field_by_name(&field.name) {
                        let list = by_field
                            .get_mut(&target.number)
                            .expect("target's own field");
                        if list.is_empty() {
                            active_fields.push(target.number);
                        }
                        list.push(SegmentPostings {
                            source: index,
                            field,
                            state: source_state,
                        });
                    }
                }
            }
            active_fields.sort_unstable();
            for &number in &active_fields {
                let field_sources = by_field.get_mut(&number).expect("active target field");
                let target = state.field_info(number).expect("target's own field");
                let merged = StMergingTermsEnum::postings(&term, field_sources, &sources, target)?;
                field_sources.clear();
                // writeTerm returns null when all documents were deleted.
                if merged.docs.is_empty() {
                    continue;
                }
                let seen = docs_seen
                    .entry(number)
                    .or_insert_with(|| FixedBitSet::new(max_doc));
                let seen_len = seen.len();
                for &(doc, _) in &merged.docs {
                    let doc = doc as usize;
                    if doc >= seen_len {
                        return Err(corrupt("mapped document exceeds target maxDoc"));
                    }
                    seen.set(doc);
                }
                let input = FieldPostingsInput {
                    field_number: number,
                    index_options: target.index_options,
                    has_payloads: target.store_payloads,
                    // Final cardinalities are computed once, when writing
                    // field metadata, as FieldMetadata.getDocCount does.
                    doc_count: 0,
                    terms: std::slice::from_ref(&merged),
                };
                postings.write_field(&input, norms, |field| terms.add_term_states(field))?;
            }
            // `nextTermForIterators`: the grouped readers stay on the old
            // term until every field's postings has been consumed.
            for &index in &grouped {
                if let Some(next) = sources[index].next_term()? {
                    queue.push(Reverse((next, index)));
                }
            }
        }
        let (ustb, ustd) = terms.finish_merge(&docs_seen)?;
        Ok(UniformSplitOutput {
            postings: postings.finish()?,
            ustb,
            ustd,
        })
    }
}

#[cfg(test)]
mod tests;
