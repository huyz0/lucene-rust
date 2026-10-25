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
            postings: postings.finish(),
            ustb,
            ustd,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_state::SegmentReadState;

    const ID: [u8; 16] = *b"sharedmergetests";
    const SUFFIX: &str = "SharedTermsUniformSplit_0";

    fn infos() -> FieldInfos {
        FieldInfos::new(vec![
            FieldInfo::new("docs", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
            FieldInfo::new("freq", 1)
                .with_index_options(IndexOptions::DocsAndFreqs)
                .with_omit_norms(true),
            FieldInfo::new("pay", 2)
                .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
                .with_store_payloads(true)
                .with_omit_norms(true),
        ])
        .unwrap()
    }

    fn state(infos: &FieldInfos) -> SegmentWriteState<'_> {
        SegmentWriteState {
            max_doc: 3,
            segment_id: &ID,
            segment_suffix: SUFFIX,
            field_infos: infos,
        }
    }

    fn read(output: &UniformSplitOutput, infos: &FieldInfos) -> StUniformSplitTermsReader {
        StUniformSplitPostingsFormat::default()
            .open(
                output.ustb.clone(),
                &output.ustd,
                &SegmentReadState {
                    max_doc: 3,
                    segment_id: &ID,
                    segment_suffix: SUFFIX,
                    field_infos: infos,
                },
            )
            .unwrap()
    }

    fn fixture(infos: &FieldInfos) -> UniformSplitOutput {
        let fields: Vec<_> = infos
            .fields
            .iter()
            .map(|field| {
                [b"a".to_vec(), b"z".to_vec()]
                    .into_iter()
                    .map(|term| TermPostings {
                        term,
                        docs: vec![(0, 1), (1, 1), (2, 1)],
                        positions: if field.index_options.subsumes_positions() {
                            vec![vec![0]; 3]
                        } else {
                            Vec::new()
                        },
                        offsets: if field.index_options.subsumes_offsets() {
                            vec![vec![(0, 1)]; 3]
                        } else {
                            Vec::new()
                        },
                        payload_lengths: if field.store_payloads {
                            vec![1; 3]
                        } else {
                            vec![]
                        },
                        payload_bytes: if field.store_payloads {
                            vec![42; 3]
                        } else {
                            vec![]
                        },
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let inputs: Vec<_> = infos
            .fields
            .iter()
            .zip(&fields)
            .map(|(field, terms)| FieldPostingsInput {
                field_number: field.number,
                index_options: field.index_options,
                has_payloads: field.store_payloads,
                doc_count: 3,
                terms,
            })
            .collect();
        StUniformSplitPostingsFormat::default()
            .write(&inputs, &[], &state(infos))
            .unwrap()
    }

    fn no_fallback() -> Result<UniformSplitOutput> {
        panic!("specialized merge unexpectedly delegated")
    }

    #[test]
    fn maps_deletes_and_multiple_sources_preserve_postings() {
        let infos = infos();
        let output = fixture(&infos);
        let reader = read(&output, &infos);
        let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
        let pos = PosInput::open(&output.postings.pos, &ID, SUFFIX).unwrap();
        let pay = PayInput::open(&output.postings.pay, &ID, SUFFIX).unwrap();
        let source = StMergeSegment {
            reader: Some(&reader),
            field_infos: &infos,
            doc_map: &[0, -1, 1],
            doc: &doc,
            pos: Some(&pos),
            pay: Some(&pay),
        };
        let second = StMergeSegment {
            doc_map: &[-1, 2, -1],
            ..source
        };
        let merged = StUniformSplitPostingsFormat::default()
            .merge(&[source, second], &[], &state(&infos), false, no_fallback)
            .unwrap();
        // Both terms in all fields have all three target documents, with
        // identical positions, offsets and payloads to the original bytes.
        assert_eq!(merged.ustb, output.ustb);
        assert_eq!(merged.ustd, output.ustd);
        assert_eq!(merged.postings.doc, output.postings.doc);
        assert_eq!(merged.postings.pos, output.postings.pos);
        assert_eq!(merged.postings.pay, output.postings.pay);
        assert_eq!(merged.postings.psm, output.postings.psm);
    }

    #[test]
    fn mixed_payload_sources_and_downgraded_offsets() {
        let full = infos();
        let a_infos = FieldInfos::new(vec![full.field_by_name("pay").unwrap().clone()]).unwrap();
        let b_infos = FieldInfos::new(vec![FieldInfo::new("pay", 17)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
            .with_omit_norms(true)])
        .unwrap();
        let a = fixture(&a_infos);
        let b = fixture(&b_infos);
        let a_reader = read(&a, &a_infos);
        let b_reader = read(&b, &b_infos);
        let a_doc = DocInput::open(&a.postings.doc, &ID, SUFFIX).unwrap();
        let b_doc = DocInput::open(&b.postings.doc, &ID, SUFFIX).unwrap();
        let a_pos = PosInput::open(&a.postings.pos, &ID, SUFFIX).unwrap();
        let b_pos = PosInput::open(&b.postings.pos, &ID, SUFFIX).unwrap();
        let a_pay = PayInput::open(&a.postings.pay, &ID, SUFFIX).unwrap();
        let target = FieldInfos::new(vec![FieldInfo::new("pay", 42)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
            .with_store_payloads(true)
            .with_omit_norms(true)])
        .unwrap();
        let target_state = SegmentWriteState {
            max_doc: 2,
            ..state(&target)
        };
        let sources = [
            StMergeSegment {
                reader: Some(&a_reader),
                field_infos: &a_infos,
                doc_map: &[0, -1, -1],
                doc: &a_doc,
                pos: Some(&a_pos),
                pay: Some(&a_pay),
            },
            StMergeSegment {
                reader: Some(&b_reader),
                field_infos: &b_infos,
                doc_map: &[-1, 1, -1],
                doc: &b_doc,
                pos: Some(&b_pos),
                pay: None,
            },
        ];
        let format = StUniformSplitPostingsFormat::default();
        let merged = format
            .merge(&sources, &[], &target_state, false, no_fallback)
            .unwrap();
        let expected_terms: Vec<_> = [b"a".to_vec(), b"z".to_vec()]
            .into_iter()
            .map(|term| TermPostings {
                term,
                docs: vec![(0, 1), (1, 1)],
                positions: vec![vec![0], vec![0]],
                payload_lengths: vec![1, 0],
                payload_bytes: vec![42],
                ..TermPostings::default()
            })
            .collect();
        let expected = format
            .write(
                &[FieldPostingsInput {
                    field_number: 42,
                    index_options: IndexOptions::DocsAndFreqsAndPositions,
                    has_payloads: true,
                    doc_count: 2,
                    terms: &expected_terms,
                }],
                &[],
                &target_state,
            )
            .unwrap();
        // The second source contributes a zero-length payload. Source
        // offsets are dropped, while the first source's payload survives.
        assert_eq!(merged.ustb, expected.ustb);
        assert_eq!(merged.ustd, expected.ustd);
        assert_eq!(merged.postings.doc, expected.postings.doc);
        assert_eq!(merged.postings.pos, expected.postings.pos);
        assert_eq!(merged.postings.pay, expected.postings.pay);
        assert_eq!(merged.postings.psm, expected.postings.psm);
    }

    #[test]
    fn invalid_maps_and_missing_streams_are_errors() {
        let infos = infos();
        let output = fixture(&infos);
        let reader = read(&output, &infos);
        let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
        let pos = PosInput::open(&output.postings.pos, &ID, SUFFIX).unwrap();
        let pay = PayInput::open(&output.postings.pay, &ID, SUFFIX).unwrap();
        let source = StMergeSegment {
            reader: Some(&reader),
            field_infos: &infos,
            doc_map: &[0, 1, 2],
            doc: &doc,
            pos: Some(&pos),
            pay: Some(&pay),
        };
        for (map, expected) in [
            (&[0, 1][..], "outside its doc map"),
            (&[0, -2, 1][..], "not increasing"),
            (&[0, 0, 1][..], "not increasing"),
            (&[1, 0, 2][..], "not increasing"),
            (&[0, 1, 3][..], "exceeds target maxDoc"),
        ] {
            let result = StUniformSplitPostingsFormat::default().merge(
                &[StMergeSegment {
                    doc_map: map,
                    ..source
                }],
                &[],
                &state(&infos),
                false,
                no_fallback,
            );
            assert!(
                matches!(result, Err(Error::Store(lucene_store::Error::Corrupted(ref msg))) if msg.contains(expected)),
                "{result:?}"
            );
        }
        let missing_pos = StUniformSplitPostingsFormat::default().merge(
            &[StMergeSegment {
                pos: None,
                ..source
            }],
            &[],
            &state(&infos),
            false,
            no_fallback,
        );
        assert!(
            matches!(missing_pos, Err(Error::Store(lucene_store::Error::Corrupted(ref msg))) if msg.contains("lacks .pos"))
        );
        let missing_pay = StUniformSplitPostingsFormat::default().merge(
            &[StMergeSegment {
                pay: None,
                ..source
            }],
            &[],
            &state(&infos),
            false,
            no_fallback,
        );
        assert!(missing_pay.is_err(), "offsets and payloads require .pay");
        let negative = SegmentWriteState {
            max_doc: -1,
            ..state(&infos)
        };
        assert!(matches!(
            StUniformSplitPostingsFormat::default().merge(
                &[source],
                &[],
                &negative,
                false,
                no_fallback,
            ),
            Err(Error::IllegalArgument(_))
        ));
    }

    #[test]
    fn empty_sources_deleted_fields_and_filtered_fields() {
        let infos = infos();
        let output = fixture(&infos);
        let reader = read(&output, &infos);
        let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
        let source = StMergeSegment {
            reader: Some(&reader),
            field_infos: &infos,
            doc_map: &[-1; 3],
            doc: &doc,
            pos: None,
            pay: None,
        };
        let empty = FieldInfos::new(vec![]).unwrap();
        let format = StUniformSplitPostingsFormat::default();
        for sources in [
            vec![],
            vec![source],
            vec![StMergeSegment {
                field_infos: &empty,
                ..source
            }],
        ] {
            // Removed target fields do not read source positions, even if
            // their source terms exist and their doc maps are empty.
            let merged = format
                .merge(&sources, &[], &state(&empty), false, no_fallback)
                .unwrap();
            assert_eq!(read(&merged, &empty).field_names().count(), 0);
        }
        let docs_only =
            FieldInfos::new(vec![infos.field_by_name("docs").unwrap().clone()]).unwrap();
        let merged = format
            .merge(&[source], &[], &state(&docs_only), false, no_fallback)
            .unwrap();
        assert_eq!(read(&merged, &docs_only).field_names().count(), 0);
        // A target can downgrade positions/frequencies to DOCS. It does
        // not need .pos/.pay, and writes frequency one for each survivor.
        let downgraded = FieldInfos::new(vec![FieldInfo::new("pay", 5)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true)])
        .unwrap();
        let merged = format
            .merge(
                &[StMergeSegment {
                    doc_map: &[0, 1, 2],
                    ..source
                }],
                &[],
                &state(&downgraded),
                false,
                no_fallback,
            )
            .unwrap();
        let reader = read(&merged, &downgraded);
        let field = reader.terms("pay").unwrap();
        assert_eq!(field.doc_count(), 3);
        assert_eq!(field.sum_doc_freq(), 6);
        assert_eq!(field.sum_total_term_freq(), 6);
    }

    #[test]
    fn fallback_is_once_before_input_validation_and_propagates_errors() {
        let infos = infos();
        let output = fixture(&infos);
        let doc = DocInput::open(&output.postings.doc, &ID, SUFFIX).unwrap();
        let source = StMergeSegment {
            reader: None,
            field_infos: &infos,
            doc_map: &[],
            doc: &doc,
            pos: None,
            pay: None,
        };
        let state = SegmentWriteState {
            max_doc: -1,
            ..state(&infos)
        };
        for (sources, sorted) in [(vec![], true), (vec![source], false)] {
            let mut calls = 0;
            let error = StUniformSplitPostingsFormat::default().merge(
                &sources,
                &[],
                &state,
                sorted,
                || {
                    calls += 1;
                    Err(Error::IllegalArgument("fallback error".into()))
                },
            );
            assert_eq!(calls, 1);
            assert!(
                matches!(error, Err(Error::IllegalArgument(ref msg)) if msg == "fallback error")
            );
        }
    }
}
