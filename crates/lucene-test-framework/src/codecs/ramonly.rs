//! `RAMOnlyPostingsFormat`: a per-instance map of postings, identified by a
//! `.id` token (`CodecUtil` header `RAMOnly`, version 0, then a vInt id).
//! There is no footer, segment id or suffix in the token. It cannot reopen
//! an index in another process or through another format instance.
//!
//! Java stores terms as strings in a `TreeMap`: UTF-16 order, including the
//! supplementary/BMP ordering difference from UTF-8. `seekCeil` retains the
//! requested term; its next call starts at the inclusive tail map, even
//! after an exact seek. These test-format quirks are preserved.
//!
//! Rust returns token bytes instead of writing a Directory. Producers share
//! the map; a retrieved field is an owned snapshot. Drop replaces close.
//! Invalid UTF-8 and malformed postings return errors; Java assumes valid
//! UTF-8 and asserts or throws on malformed input. Infallible postings
//! accessors return -1/0 before positioning where Java dereferences null.
//! IndexWriter does not select this test format. The internal RAMDoc's
//! JVM-array memory estimator has no Rust counterpart.

use std::collections::btree_map::Range;
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::postings_writer::{FieldPostingsInput, TermPostings, TermSource};
use lucene_codecs::segment_state::SegmentWriteState;
use lucene_search::reader::{
    DocIdSetIterator, PostingsEnum, PostingsFlags, Terms, TermsEnum, NO_MORE_DOCS,
};
use lucene_search::{Error, Result};
use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;
use lucene_util::fixed_bit_set::FixedBitSet;

fn key(bytes: &[u8]) -> Result<Vec<u16>> {
    Ok(std::str::from_utf8(bytes)
        .map_err(|_| Error::IllegalArgument("RAMOnly requires valid UTF-8 terms".into()))?
        .encode_utf16()
        .collect())
}

/// Java's instance-local state and atomic slot counter.
#[derive(Debug, Default)]
pub struct RamOnlyPostingsFormat {
    next_id: AtomicI32,
    state: Mutex<HashMap<i32, RamPostings>>,
}

impl RamOnlyPostingsFormat {
    pub fn name(&self) -> &'static str {
        "RAMOnly"
    }

    /// Java: `fieldsConsumer`. The slot is allocated before `write`.
    pub fn fields_consumer(&self) -> RamFieldsConsumer {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut token = Vec::new();
        codec_util::write_header(&mut token, "RAMOnly", 0);
        token.write_vint(id);
        let postings = RamPostings::default();
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, postings.clone());
        RamFieldsConsumer { token, postings }
    }

    /// Java: `fieldsProducer`; an unknown slot returns Java's null (`None`).
    /// Trailing bytes are ignored, as in Java's footerless token reader.
    ///
    /// # Errors
    /// A damaged header/version or a truncated vInt.
    pub fn fields_producer(&self, token: &[u8]) -> Result<Option<RamPostings>> {
        let mut input = SliceInput::new(token);
        codec_util::check_header(&mut input, "RAMOnly", 0, 0)?;
        let id = input.read_vint()?;
        Ok(self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&id)
            .cloned())
    }
}

/// Java: `RAMPostings`. A producer and its consumer share this state.
#[derive(Debug, Clone, Default)]
pub struct RamPostings {
    fields: Arc<Mutex<BTreeMap<Vec<u16>, Arc<RamField>>>>,
}

impl RamPostings {
    pub fn field_names(&self) -> Vec<String> {
        self.fields
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys()
            .map(|k| String::from_utf16(k).expect("field names came from Rust strings"))
            .collect()
    }
    pub fn size(&self) -> usize {
        self.fields
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
    pub fn terms(&self, name: &str) -> Option<Arc<RamField>> {
        self.fields
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&name.encode_utf16().collect::<Vec<_>>())
            .cloned()
    }
    /// Java's `checkIntegrity` is a no-op: postings have no disk bytes.
    pub fn check_integrity(&self) -> Result<()> {
        Ok(())
    }
}

/// Java: `RAMFieldsConsumer`; dropping it does not remove the map entry.
#[derive(Debug)]
pub struct RamFieldsConsumer {
    token: Vec<u8>,
    postings: RamPostings,
}

impl RamFieldsConsumer {
    pub fn token(&self) -> &[u8] {
        &self.token
    }

    /// Java: `RAMFieldsConsumer.write`. Each pair names a field by number
    /// and supplies its source dictionary. Requests Java's exact flags:
    /// PAYLOADS with payloads, NONE otherwise (even with positions indexed).
    /// The returned data depends on the source: Lucene104 returns freq 1
    /// and position -1 for NONE; RAMOnly itself ignores the requested flags.
    ///
    /// # Errors
    /// Source reader errors and the validation errors of
    /// [`Self::write_materialized`].
    pub fn write(
        &mut self,
        inputs: &[(i32, &dyn Terms)],
        state: &SegmentWriteState<'_>,
    ) -> Result<()> {
        for &(number, terms) in inputs {
            let info = state.field_infos.field_by_number(number).ok_or_else(|| {
                Error::IllegalArgument("RAMOnly field is missing from FieldInfos".into())
            })?;
            Self::validate_options(info, state.max_doc)?;
            let flags = if info.store_payloads {
                PostingsFlags::Payloads
            } else {
                PostingsFlags::None
            };
            let mut source = terms.iterator()?;
            let mut materialized = Vec::new();
            while let Some(bytes) = source.next()? {
                let mut term = TermPostings {
                    term: bytes.to_vec(),
                    ..TermPostings::default()
                };
                let mut p = source.postings(flags)?;
                loop {
                    let doc = p.next_doc()?;
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    let freq = p.freq();
                    if doc < 0 || doc >= state.max_doc || freq <= 0 {
                        return Err(Error::IllegalArgument(
                            "RAMOnly invalid doc, frequency or positions".into(),
                        ));
                    }
                    term.docs.push((doc, freq));
                    let mut positions = Vec::new();
                    for _ in 0..freq {
                        positions.push(p.next_position()?);
                        if info.store_payloads {
                            let payload = p.payload().unwrap_or_default();
                            term.payload_lengths
                                .push(u32::try_from(payload.len()).map_err(|_| {
                                    Error::IllegalArgument("RAMOnly payload length overflow".into())
                                })?);
                            term.payload_bytes.extend_from_slice(payload);
                        }
                    }
                    term.positions.push(positions);
                }
                materialized.push(term);
            }
            self.write_materialized(
                &[FieldPostingsInput {
                    field_number: number,
                    index_options: info.index_options,
                    doc_count: terms.doc_count(),
                    has_payloads: info.store_payloads,
                    terms: &materialized,
                }],
                state,
            )?;
        }
        Ok(())
    }

    fn validate_options(info: &FieldInfo, max_doc: i32) -> Result<()> {
        if info.index_options.subsumes_offsets() {
            return Err(Error::Unsupported("this codec cannot index offsets".into()));
        }
        if !info.index_options.subsumes_positions() || max_doc < 0 {
            return Err(Error::IllegalArgument(
                "RAMOnly requires positions and a nonnegative maxDoc".into(),
            ));
        }
        Ok(())
    }

    /// Copies postings already selected by a source reader. This low-level
    /// helper preserves supplied frequencies and positions; callers needing
    /// Java's source-dependent flag selection must use [`Self::write`].
    /// Java: `RAMTermsConsumer` and `RAMPostingsWriterImpl`.
    /// Norms are unused by this format. Fields' options come from FieldInfos.
    ///
    /// # Errors
    /// Offsets are unsupported, as in Java. Missing fields, invalid doc ids,
    /// frequencies, positions, payload runs or UTF-8 return an argument error
    /// instead of Java's assertion, null or array exceptions. Java's broken
    /// docs-only/frequencies-only write paths are refused explicitly.
    pub fn write_materialized<T: TermSource>(
        &mut self,
        inputs: &[FieldPostingsInput<'_, T>],
        state: &SegmentWriteState<'_>,
    ) -> Result<()> {
        let mut fields = self
            .postings
            .fields
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for input in inputs {
            let info = state
                .field_infos
                .field_by_number(input.field_number)
                .ok_or_else(|| {
                    Error::IllegalArgument("RAMOnly field is missing from FieldInfos".into())
                })?;
            Self::validate_options(info, state.max_doc)?;
            let mut field = RamField {
                info: info.clone(),
                terms: BTreeMap::new(),
                sum_total_term_freq: 0,
                sum_doc_freq: 0,
                doc_count: 0,
            };
            let mut seen = FixedBitSet::new(state.max_doc as usize);
            for source in input.terms {
                let term_key = key(source.term())?;
                if source.docs().is_empty() {
                    return Err(Error::IllegalArgument(
                        "RAMOnly term has no documents".into(),
                    ));
                }
                let mut term = RamTerm {
                    bytes: source.term().to_vec(),
                    docs: Vec::new(),
                    total_term_freq: 0,
                };
                let mut payload_at = 0usize;
                let mut occurrence = 0;
                for (i, &(doc, freq)) in source.docs().iter().enumerate() {
                    if doc < 0
                        || doc as usize >= seen.len()
                        || freq <= 0
                        || source.positions(i).len() != freq as usize
                    {
                        return Err(Error::IllegalArgument(
                            "RAMOnly invalid doc, frequency or positions".into(),
                        ));
                    }
                    seen.set(doc as usize);
                    let mut payloads = Vec::new();
                    for _ in 0..freq {
                        let payload = if info.store_payloads {
                            let len =
                                *source.payload_lengths().get(occurrence).ok_or_else(|| {
                                    Error::IllegalArgument(
                                        "RAMOnly truncated payload lengths".into(),
                                    )
                                })? as usize;
                            let end = payload_at.checked_add(len).ok_or_else(|| {
                                Error::IllegalArgument("RAMOnly payload length overflow".into())
                            })?;
                            let bytes = source
                                .payload_bytes()
                                .get(payload_at..end)
                                .ok_or_else(|| {
                                    Error::IllegalArgument("RAMOnly truncated payload bytes".into())
                                })?
                                .to_vec();
                            payload_at = end;
                            bytes
                        } else {
                            Vec::new()
                        };
                        payloads.push(payload);
                        occurrence += 1;
                    }
                    term.docs.push(RamDoc {
                        id: doc,
                        positions: source.positions(i).to_vec(),
                        payloads,
                    });
                    term.total_term_freq += i64::from(freq);
                }
                if info.store_payloads
                    && (payload_at != source.payload_bytes().len()
                        || occurrence != source.payload_lengths().len())
                {
                    return Err(Error::IllegalArgument("RAMOnly excess payload data".into()));
                }
                field.sum_doc_freq += term.docs.len() as i64;
                field.sum_total_term_freq += term.total_term_freq;
                field.terms.insert(term_key, Arc::new(term));
            }
            field.doc_count = seen.cardinality() as i32;
            fields.insert(info.name.encode_utf16().collect(), Arc::new(field));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct RamTerm {
    bytes: Vec<u8>,
    docs: Vec<RamDoc>,
    total_term_freq: i64,
}
#[derive(Debug)]
struct RamDoc {
    id: i32,
    positions: Vec<i32>,
    payloads: Vec<Vec<u8>>,
}

/// Java: `RAMField`. A `TreeMap<String, RAMTerm>` and field statistics.
#[derive(Debug, Clone)]
pub struct RamField {
    info: FieldInfo,
    terms: BTreeMap<Vec<u16>, Arc<RamTerm>>,
    sum_total_term_freq: i64,
    sum_doc_freq: i64,
    doc_count: i32,
}

impl Terms for RamField {
    fn iterator(&self) -> Result<Box<dyn TermsEnum + '_>> {
        Ok(Box::new(RamTermsEnum {
            field: self,
            current: None,
            current_bytes: Vec::new(),
            iter: None,
        }))
    }
    fn size(&self) -> i64 {
        self.terms.len() as i64
    }
    fn sum_total_term_freq(&self) -> i64 {
        self.sum_total_term_freq
    }
    fn sum_doc_freq(&self) -> i64 {
        self.sum_doc_freq
    }
    fn doc_count(&self) -> i32 {
        self.doc_count
    }
    fn has_freqs(&self) -> bool {
        true
    } // The only writable options include positions.
    fn has_offsets(&self) -> bool {
        self.info.index_options.subsumes_offsets()
    }
    fn has_positions(&self) -> bool {
        self.info.index_options.subsumes_positions()
    }
    fn has_payloads(&self) -> bool {
        self.info.store_payloads
    }
}

struct RamTermsEnum<'a> {
    field: &'a RamField,
    current: Option<Vec<u16>>,
    current_bytes: Vec<u8>,
    iter: Option<Range<'a, Vec<u16>, Arc<RamTerm>>>,
}

impl RamTermsEnum<'_> {
    fn current_term(&self) -> Result<&Arc<RamTerm>> {
        self.current
            .as_ref()
            .and_then(|k| self.field.terms.get(k))
            .ok_or_else(|| Error::IllegalState("RAMOnly enum is not on an indexed term".into()))
    }
}

impl TermsEnum for RamTermsEnum<'_> {
    fn next(&mut self) -> Result<Option<&[u8]>> {
        if self.iter.is_none() {
            self.iter = Some(match &self.current {
                None => self.field.terms.range::<Vec<u16>, _>(..),
                Some(current) => self
                    .field
                    .terms
                    .range((Bound::Included(current.clone()), Bound::Unbounded)),
            });
        }
        let Some((k, term)) = self.iter.as_mut().unwrap().next() else {
            return Ok(None);
        };
        self.current = Some(k.clone());
        self.current_bytes.clone_from(&term.bytes);
        Ok(Some(&self.current_bytes))
    }
    fn term(&self) -> Option<&[u8]> {
        self.current.as_ref().map(|_| self.current_bytes.as_slice())
    }
    fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let target_key = key(target)?;
        self.current = Some(target_key.clone());
        self.current_bytes = target.to_vec();
        self.iter = None;
        let result = if self.field.terms.contains_key(&target_key) {
            SeekStatus::Found
        } else if &target_key
            > self
                .field
                .terms
                .last_key_value()
                .ok_or_else(|| Error::IllegalState("RAMOnly seekCeil on an empty field".into()))?
                .0
        {
            SeekStatus::End
        } else {
            SeekStatus::NotFound
        };
        Ok(result)
    }
    fn doc_freq(&mut self) -> Result<i32> {
        Ok(self.current_term()?.docs.len() as i32)
    }
    fn total_term_freq(&mut self) -> Result<i64> {
        Ok(self.current_term()?.total_term_freq)
    }
    fn postings(&mut self, _flags: PostingsFlags) -> Result<Box<dyn PostingsEnum>> {
        Ok(Box::new(RamDocsEnum {
            term: self.current_term()?.clone(),
            current: None,
            next: 0,
            pos: 0,
        }))
    }
}

struct RamDocsEnum {
    term: Arc<RamTerm>,
    current: Option<usize>,
    next: usize,
    pos: usize,
}
impl RamDocsEnum {
    fn doc(&self) -> Option<&RamDoc> {
        self.current.and_then(|i| self.term.docs.get(i))
    }
}
impl DocIdSetIterator for RamDocsEnum {
    // SENTINEL: -1 before positioning; Java throws NullPointerException here.
    // After exhaustion Java retains its last RAMDoc, so doc_id does too.
    fn doc_id(&self) -> i32 {
        self.doc().map_or(-1, |d| d.id)
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.next >= self.term.docs.len() {
            return Ok(NO_MORE_DOCS);
        }
        self.current = Some(self.next);
        self.next += 1;
        self.pos = 0;
        Ok(self.doc_id())
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        loop {
            let doc = self.next_doc()?;
            if doc >= target {
                return Ok(doc);
            }
        }
    }
    fn cost(&self) -> i64 {
        self.term.docs.len() as i64
    }
}
impl PostingsEnum for RamDocsEnum {
    fn freq(&self) -> i32 {
        self.doc().map_or(0, |d| d.positions.len() as i32)
    }
    fn next_position(&mut self) -> Result<i32> {
        let value = self
            .doc()
            .and_then(|d| d.positions.get(self.pos))
            .copied()
            .ok_or_else(|| {
                Error::IllegalState("RAMOnly nextPosition called beyond frequency".into())
            })?;
        self.pos += 1;
        Ok(value)
    }
    // SENTINEL: -1 = offsets are not indexed; callers compare, never index.
    fn start_offset(&self) -> i32 {
        -1
    }
    // SENTINEL: -1 = offsets are not indexed; callers compare, never index.
    fn end_offset(&self) -> i32 {
        -1
    }
    fn payload(&self) -> Option<&[u8]> {
        self.doc()?
            .payloads
            .get(self.pos.checked_sub(1)?)
            .map(Vec::as_slice)
            .filter(|p| !p.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{FieldInfos, IndexOptions};
    use lucene_codecs::postings_writer::TermPostings;

    fn infos(options: IndexOptions, payloads: bool) -> FieldInfos {
        let mut info = FieldInfo::new("body", 0).with_index_options(options);
        info.store_payloads = payloads;
        FieldInfos::new(vec![info]).unwrap()
    }

    fn write(
        infos: &FieldInfos,
        terms: &[TermPostings],
        max_doc: i32,
        number: i32,
    ) -> Result<RamPostings> {
        let format = RamOnlyPostingsFormat::default();
        let mut consumer = format.fields_consumer();
        consumer.write_materialized(
            &[FieldPostingsInput {
                field_number: number,
                index_options: infos.fields[0].index_options,
                doc_count: 1,
                has_payloads: infos.fields[0].store_payloads,
                terms,
            }],
            &SegmentWriteState {
                max_doc,
                segment_id: &[0; 16],
                segment_suffix: "",
                field_infos: infos,
            },
        )?;
        Ok(format.fields_producer(consumer.token())?.unwrap())
    }

    fn term() -> TermPostings {
        TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![2]],
            ..TermPostings::default()
        }
    }

    #[test]
    fn positional_fields_without_payloads_and_empty_fields() {
        let infos = infos(IndexOptions::DocsAndFreqsAndPositions, false);
        let mut repeated = term();
        repeated.docs[0].1 = 2;
        repeated.positions[0] = vec![2, 4];
        let producer = write(&infos, &[repeated], 1, 0).unwrap();
        let field = producer.terms("body").unwrap();
        assert!(!field.has_payloads());
        assert!(field.has_freqs());
        assert!(field.has_positions());
        assert!(!field.has_offsets());
        let mut e = field.iterator().unwrap();
        assert_eq!(e.next().unwrap(), Some(b"a".as_slice()));
        let mut p = e.postings(PostingsFlags::None).unwrap();
        assert_eq!(p.next_doc().unwrap(), 0);
        assert!(p.payload().is_none());
        assert_eq!(p.next_position().unwrap(), 2);
        assert_eq!(p.freq(), 2);
        assert_eq!(p.next_position().unwrap(), 4);
        assert!(p.payload().is_none());
        assert_eq!(p.next_doc().unwrap(), NO_MORE_DOCS);
        // Unlike Lucene104, RAMOnly's source enum ignores NONE and returns
        // the full supplied positions. The consumer must preserve that data.
        let format = RamOnlyPostingsFormat::default();
        let mut consumer = format.fields_consumer();
        let state = SegmentWriteState {
            max_doc: 1,
            segment_id: &[0; 16],
            segment_suffix: "",
            field_infos: &infos,
        };
        consumer.write(&[(0, field.as_ref())], &state).unwrap();
        let copied = format.fields_producer(consumer.token()).unwrap().unwrap();
        let copied_field = copied.terms("body").unwrap();
        let mut e = copied_field.iterator().unwrap();
        assert_eq!(e.next().unwrap(), Some(b"a".as_slice()));
        let mut p = e.postings(PostingsFlags::None).unwrap();
        assert_eq!(p.next_doc().unwrap(), 0);
        assert_eq!(p.freq(), 2);
        assert_eq!(p.next_position().unwrap(), 2);
        assert_eq!(p.next_position().unwrap(), 4);
        assert!(consumer.write(&[(99, field.as_ref())], &state).is_err());
        assert!(consumer
            .write(
                &[(0, field.as_ref())],
                &SegmentWriteState {
                    max_doc: 0,
                    ..state
                }
            )
            .is_err());
        let empty = write(&infos, &[], 0, 0).unwrap();
        let field = empty.terms("body").unwrap();
        assert_eq!(
            (
                field.size(),
                field.doc_count(),
                field.sum_doc_freq(),
                field.sum_total_term_freq()
            ),
            (0, 0, 0, 0)
        );
        let mut e = field.iterator().unwrap();
        assert!(e.next().unwrap().is_none());
        assert!(e.try_seek_ceil(b"x").is_err());
        assert_eq!(e.term(), Some(b"x".as_slice()));
    }

    #[test]
    fn malformed_inputs_fail_instead_of_asserting_or_indexing_out_of_bounds() {
        let valid = infos(IndexOptions::DocsAndFreqsAndPositions, false);
        assert!(write(&valid, &[term()], 1, 99).is_err());
        assert!(write(&valid, &[term()], -1, 0).is_err());
        for option in [
            IndexOptions::Docs,
            IndexOptions::DocsAndFreqs,
            IndexOptions::DocsAndCustomFreqs,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        ] {
            assert!(write(&infos(option, false), &[term()], 1, 0).is_err());
        }
        let mut bad = term();
        bad.term = vec![255];
        assert!(write(&valid, &[bad], 1, 0).is_err());
        let mut bad = term();
        bad.docs.clear();
        assert!(write(&valid, &[bad], 1, 0).is_err());
        for doc in [-1, 1] {
            let mut bad = term();
            bad.docs[0].0 = doc;
            assert!(write(&valid, &[bad], 1, 0).is_err());
        }
        let mut bad = term();
        bad.docs[0].1 = 0;
        assert!(write(&valid, &[bad], 1, 0).is_err());
        let mut bad = term();
        bad.positions[0].clear();
        assert!(write(&valid, &[bad], 1, 0).is_err());
        let payloads = infos(IndexOptions::DocsAndFreqsAndPositions, true);
        assert!(write(&payloads, &[term()], 1, 0).is_err());
        let mut bad = term();
        bad.payload_lengths = vec![2];
        bad.payload_bytes = vec![1];
        assert!(write(&payloads, &[bad], 1, 0).is_err());
        let mut bad = term();
        bad.payload_lengths = vec![0, 0];
        assert!(write(&payloads, &[bad], 1, 0).is_err());
        let mut bad = term();
        bad.payload_lengths = vec![0];
        bad.payload_bytes = vec![1];
        assert!(write(&payloads, &[bad], 1, 0).is_err());
    }

    #[test]
    fn slots_are_atomic_and_live_for_the_formats_lifetime() {
        let format = RamOnlyPostingsFormat::default();
        let tokens = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..16)
                .map(|_| scope.spawn(|| format.fields_consumer().token().to_vec()))
                .collect();
            jobs.into_iter()
                .map(|j| j.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut distinct = tokens.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), 16);
        for token in tokens {
            assert_eq!(format.fields_producer(&token).unwrap().unwrap().size(), 0);
        }
        format.next_id.store(i32::MAX, Ordering::Relaxed);
        let high = format.fields_consumer();
        let negative = format.fields_consumer();
        assert_ne!(high.token(), negative.token());
        assert!(format.fields_producer(negative.token()).unwrap().is_some());
    }
}
