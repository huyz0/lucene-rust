//! Port of `DirectPostingsFormat`: `Lucene104PostingsFormat`'s files,
//! loaded whole into memory on open -- every term's bytes in one array,
//! every term's postings as `int[]`s, and skip pointers over runs of terms
//! sharing a prefix that `Terms.intersect` jumps along.

use std::collections::BTreeMap;
use std::fmt;

use lucene_store::codec_util::{self, FOOTER_LENGTH};
use lucene_util::automaton::{
    AutomatonType, ByteRunnableRef, CompiledAutomaton, Transition, TransitionAccessor,
};

use super::{corrupt, Error, Result};
use crate::blocktree::{self, FieldTerms, SeekStatus};
use crate::field_infos::IndexOptions;
use crate::postings::{DocInput, PayInput, PosInput, NO_MORE_DOCS};
use crate::postings_writer::{
    self, FieldNorms, FieldPostingsInput, Output, TermSource, WriteOptions,
};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// `DEFAULT_MIN_SKIP_COUNT`.
pub const DEFAULT_MIN_SKIP_COUNT: i32 = 8;
/// `DEFAULT_LOW_FREQ_CUTOFF`.
pub const DEFAULT_LOW_FREQ_CUTOFF: i32 = 32;

/// `DirectPostingsFormat`: writes exactly what `Lucene104PostingsFormat`
/// writes, and reads it back into [`DirectFields`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectPostingsFormat {
    min_skip_count: i32,
    low_freq_cutoff: i32,
}

impl Default for DirectPostingsFormat {
    fn default() -> Self {
        Self::new(DEFAULT_MIN_SKIP_COUNT, DEFAULT_LOW_FREQ_CUTOFF)
    }
}

/// One `Lucene104PostingsFormat` instance's files, as [`DirectPostingsFormat`]
/// opens them. `pos`/`pay` are `None` when no field of the instance indexes
/// positions (or payloads/offsets), as on disk.
#[derive(Debug, Clone, Copy)]
pub struct DirectFiles<'a> {
    /// `.tim`.
    pub tim: &'a [u8],
    /// `.tip`.
    pub tip: &'a [u8],
    /// `.tmd`.
    pub tmd: &'a [u8],
    /// `.doc`.
    pub doc: &'a [u8],
    /// `.pos`.
    pub pos: Option<&'a [u8]>,
    /// `.pay`.
    pub pay: Option<&'a [u8]>,
}

impl DirectPostingsFormat {
    /// `getName()`.
    pub const NAME: &'static str = "Direct";

    /// `DirectPostingsFormat(minSkipCount, lowFreqCutoff)`: a skip pointer
    /// goes down once `min_skip_count` terms in a row share a prefix; terms
    /// with `docFreq <= low_freq_cutoff` keep their postings in one `int[]`.
    pub fn new(min_skip_count: i32, low_freq_cutoff: i32) -> Self {
        DirectPostingsFormat {
            min_skip_count,
            low_freq_cutoff,
        }
    }

    /// `toString()`.
    pub fn name(&self) -> &'static str {
        Self::NAME
    }

    /// `fieldsConsumer(state)`: `Lucene104PostingsFormat`'s.
    ///
    /// # Errors
    /// What the postings writer rejects in `inputs`.
    pub fn write<T: TermSource>(
        &self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<Output> {
        Ok(postings_writer::write_fields_with_options(
            inputs,
            norms,
            &state.postings_options(WriteOptions::default()),
            state.segment_id,
            state.segment_suffix,
        )?)
    }

    /// `fieldsProducer(state)` outside a merge: `checkIntegrity()` over every
    /// file, then every field loaded into a [`DirectField`].
    ///
    /// # Errors
    /// A file whose header, footer or checksum is wrong, a terms dictionary
    /// or postings that do not decode.
    pub fn open(
        &self,
        files: &DirectFiles<'_>,
        state: &SegmentReadState<'_>,
    ) -> Result<DirectFields> {
        DirectFields::open(files, state, self.min_skip_count, self.low_freq_cutoff)
    }
}

/// `checksumEntireFile`.
fn check_integrity(buf: &[u8]) -> Result<()> {
    let footer_start = buf
        .len()
        .checked_sub(FOOTER_LENGTH)
        .ok_or_else(|| corrupt(format!("file too short for a footer: {}", buf.len())))?;
    codec_util::check_whole_file_footer(buf, footer_start)?;
    Ok(())
}

/// `DirectFields`: every field of the instance, by name.
#[derive(Debug)]
pub struct DirectFields {
    fields: BTreeMap<String, DirectField>,
}

impl DirectFields {
    fn open(
        files: &DirectFiles<'_>,
        state: &SegmentReadState<'_>,
        min_skip_count: i32,
        low_freq_cutoff: i32,
    ) -> Result<DirectFields> {
        let (id, suffix) = (state.segment_id, state.segment_suffix);
        let fields = blocktree::open(
            files.tim,
            files.tip,
            files.tmd,
            state.field_infos,
            id,
            suffix,
            state.max_doc,
        )?;
        // `postings.checkIntegrity()`.
        for buf in [
            Some(files.tim),
            Some(files.tip),
            Some(files.doc),
            files.pos,
            files.pay,
        ]
        .into_iter()
        .flatten()
        {
            check_integrity(buf)?;
        }
        let doc_in = DocInput::open(files.doc, id, suffix)?;
        let pos_in = files
            .pos
            .map(|p| PosInput::open(p, id, suffix))
            .transpose()?;
        let pay_in = files
            .pay
            .map(|p| PayInput::open(p, id, suffix))
            .transpose()?;
        let inputs = Inputs {
            doc: &doc_in,
            pos: pos_in.as_ref(),
            pay: pay_in.as_ref(),
        };
        let mut out = BTreeMap::new();
        for (name, terms) in fields.iter_fields() {
            let field = DirectField::load(terms, &inputs, min_skip_count, low_freq_cutoff)?;
            out.insert(name.to_string(), field);
        }
        Ok(DirectFields { fields: out })
    }

    /// `iterator()`: the field names, in order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// `terms(field)`.
    pub fn terms(&self, field: &str) -> Option<&DirectField> {
        self.fields.get(field)
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.fields.len()
    }
}

impl fmt::Display for DirectFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DirectFields(fields={})", self.fields.len())
    }
}

struct Inputs<'i, 'b> {
    doc: &'i DocInput<'b>,
    pos: Option<&'i PosInput<'b>>,
    pay: Option<&'i PayInput<'b>>,
}

/// `TermAndSkip`'s two shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DirectTerm {
    /// `LowFreqTerm`: `doc, [freq, [pos, [start, end], [payloadLength]]*]*`
    /// in one array, the payloads' bytes concatenated.
    Low {
        postings: Vec<i32>,
        payloads: Vec<u8>,
        doc_freq: i32,
        total_term_freq: i32,
    },
    /// `HighFreqTerm`: one array per statistic; `positions[doc]` holds
    /// `pos` (or `pos, start, end` with offsets) per occurrence, and a
    /// `None` payload is Java's `null`.
    High {
        doc_ids: Vec<i32>,
        freqs: Option<Vec<i32>>,
        positions: Option<Vec<Vec<i32>>>,
        payloads: Option<Vec<Vec<Option<Vec<u8>>>>>,
        total_term_freq: i64,
    },
}

impl DirectTerm {
    fn doc_freq(&self) -> i32 {
        match self {
            DirectTerm::Low { doc_freq, .. } => *doc_freq,
            // At most `maxDoc` documents.
            DirectTerm::High { doc_ids, .. } => doc_ids.len() as i32,
        }
    }

    fn total_term_freq(&self) -> i64 {
        match self {
            DirectTerm::Low {
                total_term_freq, ..
            } => i64::from(*total_term_freq),
            DirectTerm::High {
                total_term_freq, ..
            } => *total_term_freq,
        }
    }
}

/// `DirectField`: one field's terms and postings, in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectField {
    term_bytes: Vec<u8>,
    /// `numTerms + 1` entries: term `i` is `term_bytes[offsets[i]..offsets[i + 1]]`
    /// (Java's `int[]`: half the cache lines a binary search touches).
    term_offsets: Vec<u32>,
    skips: Vec<i32>,
    skip_offsets: Vec<usize>,
    terms: Vec<DirectTerm>,
    has_freq: bool,
    has_pos: bool,
    has_offsets: bool,
    has_payloads: bool,
    sum_total_term_freq: i64,
    doc_count: i32,
    sum_doc_freq: i64,
}

/// The constructor's state: `setSkips`/`finishSkips`/`saveSkip`'s
/// `sameCounts`, `skipCount` and each term's own skips.
struct SkipBuilder {
    same_counts: Vec<i32>,
    term_skips: Vec<Vec<i32>>,
    skip_count: usize,
    min_skip_count: i32,
}

impl SkipBuilder {
    /// `setSkips(termOrd, termBytes)`.
    // ARITH: `term_ord >= 1` where `term_ord - 1` is taken; `same_counts`
    // entries count terms of a field (at most `i32::MAX`); offsets index
    // `term_bytes`.
    #[allow(clippy::arithmetic_side_effects)]
    fn set_skips(&mut self, term_ord: usize, offsets: &[usize], bytes: &[u8]) {
        let term_length = offsets[term_ord + 1] - offsets[term_ord];
        if self.same_counts.len() < term_length {
            self.same_counts.resize(term_length, 0);
        }
        if term_ord > 0 {
            let last_term_length = offsets[term_ord] - offsets[term_ord - 1];
            let limit = term_length.min(last_term_length);
            let mut last_term_offset = offsets[term_ord - 1];
            let mut term_offset = offsets[term_ord];
            let mut i = 0;
            while i < limit {
                let same = bytes[last_term_offset] == bytes[term_offset];
                last_term_offset += 1;
                term_offset += 1;
                if same {
                    self.same_counts[i] += 1;
                    i += 1;
                } else {
                    while i < limit {
                        if self.same_counts[i] >= self.min_skip_count {
                            // Write a skip pointer to the start of this run.
                            self.save_skip(term_ord, self.same_counts[i]);
                        }
                        self.same_counts[i] = 1;
                        i += 1;
                    }
                    break;
                }
            }
            while i < last_term_length {
                if self.same_counts[i] >= self.min_skip_count {
                    self.save_skip(term_ord, self.same_counts[i]);
                }
                self.same_counts[i] = 0;
                i += 1;
            }
            for count in &mut self.same_counts[limit..term_length] {
                *count = 1;
            }
        } else {
            for count in &mut self.same_counts[..term_length] {
                *count += 1;
            }
        }
    }

    /// `finishSkips()`.
    fn finish_skips(&mut self, offsets: &[usize]) {
        let count = self.term_skips.len();
        if let (Some(&end), Some(&start)) = (
            offsets.get(count),
            count.checked_sub(1).and_then(|c| offsets.get(c)),
        ) {
            for i in 0..end.saturating_sub(start) {
                let same = self.same_counts[i];
                if same >= self.min_skip_count {
                    self.save_skip(count, same);
                }
            }
        }
        // Reverse the skips so they are ordered from longest prefix to
        // shortest.
        for skips in &mut self.term_skips {
            skips.reverse();
        }
    }

    /// `saveSkip(ord, backCount)`.
    // ARITH: a run of `back_count` equal-prefix terms ends at `ord`, so
    // `back_count <= ord`; `skip_count` counts entries of a `Vec`.
    #[allow(clippy::arithmetic_side_effects)]
    fn save_skip(&mut self, ord: usize, back_count: i32) {
        let term = ord - back_count as usize;
        self.skip_count += 1;
        // Ords are below the field's term count, an `i32`.
        self.term_skips[term].push(ord as i32);
    }
}

impl DirectField {
    // ARITH: `count` stays below `num_terms + 1`; `term_offset` sums the
    // lengths of terms held in memory.
    #[allow(clippy::arithmetic_side_effects)]
    fn load(
        terms_in: &FieldTerms,
        inputs: &Inputs<'_, '_>,
        min_skip_count: i32,
        low_freq_cutoff: i32,
    ) -> Result<DirectField> {
        let index_options = terms_in.index_options();
        let has_freq = index_options != IndexOptions::Docs;
        let has_pos = index_options.subsumes_positions();
        let has_offsets = index_options.subsumes_offsets();
        let has_payloads = terms_in.has_payloads();
        let num_terms = usize::try_from(terms_in.num_terms)
            .ok()
            .filter(|&n| i32::try_from(n).is_ok())
            .ok_or_else(|| corrupt(format!("invalid numTerms: {}", terms_in.num_terms)))?;
        let pos_in = match (has_pos, inputs.pos) {
            (false, _) => None,
            (true, Some(pos)) => Some(pos),
            (true, None) => return Err(corrupt("a field indexes positions but there is no .pos")),
        };
        let mut term_bytes = Vec::new();
        let mut term_offsets = vec![0usize; 1];
        let mut terms = Vec::new();
        let mut skips = SkipBuilder {
            same_counts: vec![0; 10],
            term_skips: Vec::new(),
            skip_count: 0,
            min_skip_count,
        };
        let mut terms_enum = terms_in.iter();
        while let Some((term, stats)) = terms_enum.try_next()? {
            if terms.len() == num_terms {
                return Err(corrupt(format!("more terms than numTerms={num_terms}")));
            }
            term_bytes.extend_from_slice(term);
            let (postings, positions) = match pos_in {
                Some(pos) => {
                    let (p, positions) = terms_enum
                        .try_current_postings_and_positions(Some(inputs.doc), pos, inputs.pay)?
                        .ok_or_else(|| corrupt("no postings on the current term"))?;
                    (p, Some(positions))
                }
                None => (
                    terms_enum
                        .try_current_postings(Some(inputs.doc))?
                        .ok_or_else(|| corrupt("no postings on the current term"))?,
                    None,
                ),
            };
            term_offsets.push(term_bytes.len());
            let freq = |i: usize| -> Result<i32> {
                postings
                    .freqs
                    .get(i)
                    .copied()
                    .ok_or_else(|| corrupt("fewer freqs than docs"))
            };
            let occurrences = |i: usize| -> Result<&[crate::postings::Position]> {
                positions
                    .as_ref()
                    .and_then(|p| p.get(i))
                    .map(Vec::as_slice)
                    .ok_or_else(|| corrupt("fewer positions than docs"))
            };
            let ent = if stats.doc_freq <= low_freq_cutoff {
                let mut ints = Vec::new();
                let mut payloads = Vec::new();
                for (i, &doc) in postings.docs.iter().enumerate() {
                    ints.push(doc);
                    if has_freq {
                        ints.push(freq(i)?);
                        if has_pos {
                            for occ in occurrences(i)? {
                                ints.push(occ.position);
                                if has_offsets {
                                    ints.push(occ.start_offset);
                                    ints.push(occ.end_offset);
                                }
                                if has_payloads {
                                    // A payload is at most a few bytes of one
                                    // position, read whole into memory.
                                    ints.push(occ.payload.len() as i32);
                                    payloads.extend_from_slice(&occ.payload);
                                }
                            }
                        }
                    }
                }
                DirectTerm::Low {
                    postings: ints,
                    payloads,
                    doc_freq: stats.doc_freq,
                    // Java's `(int) totalTermFreq`.
                    total_term_freq: stats.total_term_freq as i32,
                }
            } else {
                let mut freqs = has_freq.then(Vec::new);
                let mut all_positions = has_pos.then(Vec::new);
                let mut all_payloads = (has_pos && has_payloads).then(Vec::new);
                for i in 0..postings.docs.len() {
                    if let Some(freqs) = &mut freqs {
                        freqs.push(freq(i)?);
                    }
                    if let Some(all_positions) = &mut all_positions {
                        let occs = occurrences(i)?;
                        let mut ints = Vec::with_capacity(occs.len());
                        for occ in occs {
                            ints.push(occ.position);
                            if has_offsets {
                                ints.push(occ.start_offset);
                                ints.push(occ.end_offset);
                            }
                        }
                        all_positions.push(ints);
                        if let Some(all_payloads) = &mut all_payloads {
                            all_payloads.push(
                                occs.iter()
                                    .map(|o| (!o.payload.is_empty()).then(|| o.payload.clone()))
                                    .collect(),
                            );
                        }
                    }
                }
                DirectTerm::High {
                    doc_ids: postings.docs,
                    freqs,
                    positions: all_positions,
                    payloads: all_payloads,
                    total_term_freq: stats.total_term_freq,
                }
            };
            terms.push(ent);
            skips.term_skips.push(Vec::new());
            skips.set_skips(terms.len() - 1, &term_offsets, &term_bytes);
        }
        if terms.len() != num_terms {
            return Err(corrupt(format!(
                "{} terms, numTerms={num_terms}",
                terms.len()
            )));
        }
        skips.finish_skips(&term_offsets);
        // Java's `termOffsets` is an `int[]`: at most 2.1GB of term bytes.
        let term_offsets = term_offsets
            .into_iter()
            .map(|o| u32::try_from(o).ok().filter(|&o| o <= i32::MAX as u32))
            .collect::<Option<Vec<u32>>>()
            .ok_or_else(|| corrupt("more than 2.1GB of term bytes in one field"))?;
        let mut flat = Vec::with_capacity(skips.skip_count);
        let mut skip_offsets = Vec::with_capacity(num_terms + 1);
        for term_skips in &skips.term_skips {
            skip_offsets.push(flat.len());
            flat.extend_from_slice(term_skips);
        }
        skip_offsets.push(flat.len());
        Ok(DirectField {
            term_bytes,
            term_offsets,
            skips: flat,
            skip_offsets,
            terms,
            has_freq,
            has_pos,
            has_offsets,
            has_payloads,
            sum_total_term_freq: terms_in.sum_total_term_freq,
            doc_count: terms_in.doc_count,
            sum_doc_freq: terms_in.sum_doc_freq,
        })
    }

    /// Term `ord`'s bytes.
    // ARITH: `ord < terms.len()`, and `term_offsets` has one more entry.
    #[allow(clippy::arithmetic_side_effects)]
    fn term(&self, ord: usize) -> &[u8] {
        &self.term_bytes[self.offset(ord)..self.offset(ord + 1)]
    }

    /// `compare(ord, other)`: Java's byte loop, which beats a `memcmp`
    /// call on the short terms a binary search mostly compares.
    fn compare(&self, ord: usize, other: &[u8]) -> std::cmp::Ordering {
        let term = self.term(ord);
        for (a, b) in term.iter().zip(other) {
            if a != b {
                return a.cmp(b);
            }
        }
        term.len().cmp(&other.len())
    }

    /// `termOffsets[i]`.
    fn offset(&self, i: usize) -> usize {
        self.term_offsets[i] as usize
    }

    /// The number of terms, Java's `terms.length` (an `int`).
    fn len(&self) -> i32 {
        self.terms.len() as i32
    }

    /// `size()`.
    pub fn size(&self) -> i64 {
        self.terms.len() as i64
    }

    /// `getSumTotalTermFreq()`.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.sum_total_term_freq
    }

    /// `getSumDocFreq()`.
    pub fn sum_doc_freq(&self) -> i64 {
        self.sum_doc_freq
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        self.doc_count
    }

    /// `hasFreqs()`.
    pub fn has_freqs(&self) -> bool {
        self.has_freq
    }

    /// `hasPositions()`.
    pub fn has_positions(&self) -> bool {
        self.has_pos
    }

    /// `hasOffsets()`.
    pub fn has_offsets(&self) -> bool {
        self.has_offsets
    }

    /// `hasPayloads()`.
    pub fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    /// `iterator()`.
    pub fn iter(&self) -> DirectTermsEnum<'_> {
        DirectTermsEnum {
            field: self,
            term_ord: -1,
        }
    }

    /// `intersect(compiled, startTerm)`: the terms `compiled` accepts, after
    /// `start_term` when given.
    ///
    /// # Errors
    /// [`Error::Unsupported`] unless `compiled` is
    /// [`AutomatonType::NORMAL`], as Java throws.
    pub fn intersect<'a>(
        &'a self,
        compiled: &'a CompiledAutomaton,
        start_term: Option<&[u8]>,
    ) -> Result<DirectIntersectTermsEnum<'a>> {
        if compiled.automaton_type != AutomatonType::NORMAL {
            return Err(Error::Unsupported(
                "please use CompiledAutomaton.getTermsEnum instead",
            ));
        }
        let (Some(runnable), Some(accessor)) =
            (compiled.byte_runnable(), compiled.get_transition_accessor())
        else {
            return Err(Error::Unsupported(
                "a NORMAL automaton without its runnable",
            ));
        };
        DirectIntersectTermsEnum::new(
            self,
            runnable,
            accessor,
            compiled.common_suffix_ref.as_deref(),
            start_term,
        )
    }

    /// `postings(reuse, flags)` on term `ord`, shared by both enums: the
    /// `PostingsEnum` class Java picks for the field and `positions`
    /// (`featureRequested(flags, POSITIONS)`).
    fn postings(&self, ord: usize, positions: bool) -> DirectPostingsEnum<'_> {
        match &self.terms[ord] {
            DirectTerm::Low {
                postings, payloads, ..
            } => {
                if !self.has_freq {
                    DirectPostingsEnum::LowNoTf(LowFreqDocsEnumNoTf::new(postings))
                } else if !self.has_pos {
                    DirectPostingsEnum::LowNoPos(LowFreqDocsEnumNoPos::new(postings))
                } else if positions {
                    DirectPostingsEnum::LowPostings(LowFreqPostingsEnum::new(
                        postings,
                        payloads,
                        self.has_offsets,
                        self.has_payloads,
                    ))
                } else {
                    let pos_len = match (self.has_offsets, self.has_payloads) {
                        (true, true) => 4,
                        (true, false) => 3,
                        (false, true) => 2,
                        (false, false) => 1,
                    };
                    DirectPostingsEnum::LowDocs(LowFreqDocsEnum::new(postings, pos_len))
                }
            }
            DirectTerm::High {
                doc_ids,
                freqs,
                positions: term_positions,
                payloads,
                ..
            } => match term_positions {
                Some(term_positions) if positions => {
                    DirectPostingsEnum::HighPostings(HighFreqPostingsEnum::new(
                        doc_ids,
                        freqs.as_deref().unwrap_or(&[]),
                        term_positions,
                        payloads.as_deref(),
                        self.has_offsets,
                    ))
                }
                _ => DirectPostingsEnum::HighDocs(HighFreqDocsEnum::new(doc_ids, freqs.as_deref())),
            },
        }
    }
}

impl fmt::Display for DirectField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DirectTerms(terms={},postings={},positions={},docs={})",
            self.terms.len(),
            self.sum_doc_freq,
            self.sum_total_term_freq,
            self.doc_count
        )
    }
}

/// `DirectTermsEnum`: a cursor over the field's terms, by ord.
#[derive(Debug, Clone)]
pub struct DirectTermsEnum<'a> {
    field: &'a DirectField,
    term_ord: i32,
}

impl<'a> DirectTermsEnum<'a> {
    /// The current ord as an index, when it names a term.
    fn current_ord(&self) -> Option<usize> {
        usize::try_from(self.term_ord)
            .ok()
            .filter(|&o| o < self.field.terms.len())
    }

    /// `next()`.
    // A cursor that also seeks, as Java's `TermsEnum`, not an `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&'a [u8]> {
        if self.term_ord < self.field.len() {
            // ARITH: below `len()`, an `i32`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.term_ord += 1;
            }
        }
        self.term()
    }

    /// `findTerm(term)`: `Ok(ord)` of the term, or `Err(insertion point)`.
    // ARITH: `low <= high + 1 <= len`, and `mid >= low >= 0` where
    // `mid - 1` is taken.
    #[allow(clippy::arithmetic_side_effects)]
    fn find_term(&self, term: &[u8]) -> std::result::Result<usize, usize> {
        let mut low = 0usize;
        let mut high = self.field.terms.len();
        while low < high {
            let mid = (low + high) >> 1;
            match self.field.compare(mid, term) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(low)
    }

    /// `seekCeil(term)`. On [`SeekStatus::End`] the enum stays where it was,
    /// as Java's does.
    pub fn seek_ceil(&mut self, target: &[u8]) -> SeekStatus {
        match self.find_term(target) {
            Ok(ord) => {
                self.term_ord = ord as i32;
                SeekStatus::Found
            }
            Err(ord) if ord == self.field.terms.len() => SeekStatus::End,
            Err(ord) => {
                self.term_ord = ord as i32;
                SeekStatus::NotFound
            }
        }
    }

    /// `seekExact(term)`.
    pub fn seek_exact(&mut self, target: &[u8]) -> bool {
        match self.find_term(target) {
            Ok(ord) => {
                self.term_ord = ord as i32;
                true
            }
            Err(_) => false,
        }
    }

    /// `seekExact(ord)`.
    ///
    /// # Errors
    /// An ord outside `[0, size())`, where Java indexes past its arrays.
    pub fn seek_exact_ord(&mut self, ord: i64) -> Result<()> {
        let ord = i32::try_from(ord)
            .ok()
            .filter(|&o| o >= 0 && o < self.field.len())
            .ok_or(Error::Unsupported("seekExact(ord) out of range"))?;
        self.term_ord = ord;
        Ok(())
    }

    /// `term()`: `None` before the first term and after the last.
    pub fn term(&self) -> Option<&'a [u8]> {
        self.current_ord().map(|o| self.field.term(o))
    }

    /// `ord()`.
    pub fn ord(&self) -> i64 {
        i64::from(self.term_ord)
    }

    /// `docFreq()`, `0` off a term.
    pub fn doc_freq(&self) -> i32 {
        self.current_ord()
            .map_or(0, |o| self.field.terms[o].doc_freq())
    }

    /// `totalTermFreq()`, `0` off a term.
    pub fn total_term_freq(&self) -> i64 {
        self.current_ord()
            .map_or(0, |o| self.field.terms[o].total_term_freq())
    }

    /// `postings(reuse, flags)`, `positions` being
    /// `featureRequested(flags, POSITIONS)`; `None` off a term.
    pub fn postings(&self, positions: bool) -> Option<DirectPostingsEnum<'a>> {
        self.current_ord()
            .map(|o| self.field.postings(o, positions))
    }
}

/// `DirectIntersectTermsEnum.State`.
#[derive(Debug, Clone, Copy, Default)]
struct State {
    change_ord: i32,
    state: i32,
    transition_upto: i32,
    transition_count: i32,
    transition_max: i32,
    transition_min: i32,
    transition: Transition,
}

/// `DirectIntersectTermsEnum`: walks the terms against the automaton,
/// jumping along the skip pointers past runs the automaton rejects.
pub struct DirectIntersectTermsEnum<'a> {
    field: &'a DirectField,
    runnable: ByteRunnableRef<'a>,
    accessor: &'a dyn TransitionAccessor,
    common_suffix: Option<&'a [u8]>,
    term_ord: i32,
    states: Vec<State>,
    state_upto: usize,
    on_term: bool,
}

impl fmt::Debug for DirectIntersectTermsEnum<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DirectIntersectTermsEnum")
            .field("term_ord", &self.term_ord)
            .field("state_upto", &self.state_upto)
            .finish()
    }
}

/// `termBytes[i] & 0xFF`; Java reads past a term into the next one (and
/// throws past the last), which this reads as `0`.
fn byte_at(bytes: &[u8], i: usize) -> i32 {
    bytes.get(i).map_or(0, |&b| i32::from(b))
}

/// The outcome of the constructor's `startTerm` walk.
enum Start {
    /// `return` from inside the label loop.
    Done,
    /// Fell out of the loop: the final `termOrd >= 0` check runs.
    Check,
}

impl<'a> DirectIntersectTermsEnum<'a> {
    fn new(
        field: &'a DirectField,
        runnable: ByteRunnableRef<'a>,
        accessor: &'a dyn TransitionAccessor,
        common_suffix: Option<&'a [u8]>,
        start_term: Option<&[u8]>,
    ) -> Result<Self> {
        let mut first = State {
            change_ord: field.len(),
            state: 0,
            transition_upto: -1,
            transition_max: -1,
            ..State::default()
        };
        first.transition_count = accessor.get_num_transitions(0);
        accessor.init_transition(0, &mut first.transition);
        let mut e = DirectIntersectTermsEnum {
            field,
            runnable,
            accessor,
            common_suffix,
            term_ord: -1,
            states: vec![first],
            state_upto: 0,
            on_term: false,
        };
        if let Some(start) = start_term {
            e.seek_start(start)?;
        }
        Ok(e)
    }

    /// The constructor's `startTerm != null` block.
    // ARITH: `term_ord` stays within `[-1, len]` and `state_upto` within the
    // pushed states, as in Java; offsets index `term_bytes`.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_start(&mut self, start: &[u8]) -> Result<()> {
        let mut skip_upto = 0usize;
        let check = if start.is_empty() {
            if !self.field.terms.is_empty() && self.field.term_offsets.get(1) == Some(&0) {
                self.term_ord = 0;
            }
            Start::Check
        } else {
            self.term_ord += 1;
            self.walk_start(start, &mut skip_upto)?
        };
        if let Start::Check = check {
            if let Ok(ord) = usize::try_from(self.term_ord) {
                if ord < self.field.terms.len() && self.field.term(ord) != start {
                    self.state_upto -= skip_upto;
                    self.term_ord -= 1;
                }
            }
        }
        Ok(())
    }

    /// `nextLabel:` over `startTerm`'s bytes.
    // ARITH: as `seek_start`.
    #[allow(clippy::arithmetic_side_effects)]
    fn walk_start(&mut self, start: &[u8], skip_upto: &mut usize) -> Result<Start> {
        let f = self.field;
        let n = f.len();
        'next_label: for (i, &b) in start.iter().enumerate() {
            let label = i32::from(b);
            let st = self
                .states
                .get_mut(i)
                .ok_or_else(|| corrupt("intersect start state out of range"))?;
            while label > st.transition_max {
                st.transition_upto += 1;
                if st.transition_upto >= st.transition_count {
                    // All transitions compare less than the required label.
                    break;
                }
                self.accessor.get_next_transition(&mut st.transition);
                st.transition_min = st.transition.min;
                st.transition_max = st.transition.max;
            }

            // Skip forwards until we find a term matching the label at this
            // position.
            while self.term_ord < n {
                let ord = self.term_ord as usize;
                let skip_offset = f.skip_offsets[ord];
                let num_skips = f.skip_offsets[ord + 1] - skip_offset;
                let term_offset = f.offset(ord);
                let term_length = f.offset(ord + 1) - term_offset;
                if self.term_ord == self.states[self.state_upto].change_ord {
                    self.state_upto -= 1;
                    self.term_ord -= 1;
                    return Ok(Start::Done);
                }
                if term_length == i {
                    self.term_ord += 1;
                    *skip_upto = 0;
                } else if label < byte_at(&f.term_bytes, term_offset + i) {
                    self.term_ord -= 1;
                    self.state_upto = self
                        .state_upto
                        .checked_sub(*skip_upto)
                        .ok_or_else(|| corrupt("intersect popped past its first state"))?;
                    return Ok(Start::Done);
                } else if label == byte_at(&f.term_bytes, term_offset + i) {
                    if *skip_upto < num_skips {
                        let next_state = self
                            .runnable
                            .step(self.states[self.state_upto].state, label);
                        // Java asserts the automaton accepts `startTerm`; a
                        // dead state would index before its arrays.
                        if next_state < 0 {
                            return Err(Error::Unsupported(
                                "the automaton does not accept startTerm",
                            ));
                        }
                        let change_ord = f.skips[skip_offset + *skip_upto];
                        *skip_upto += 1;
                        self.push(next_state, change_ord);
                        continue 'next_label;
                    }
                    // Index exhausted: just scan now (the number of scans
                    // required will be less than the minSkipCount).
                    while self.term_ord < n && f.term(self.term_ord as usize) <= start {
                        self.term_ord += 1;
                    }
                    self.term_ord -= 1;
                    self.state_upto = self
                        .state_upto
                        .checked_sub(*skip_upto)
                        .ok_or_else(|| corrupt("intersect popped past its first state"))?;
                    return Ok(Start::Done);
                } else {
                    if *skip_upto < num_skips {
                        self.term_ord = f.skips[skip_offset + *skip_upto];
                    } else {
                        self.term_ord += 1;
                    }
                    *skip_upto = 0;
                }
            }

            // startTerm is >= last term so enum will not return any terms.
            self.term_ord -= 1;
            return Ok(Start::Done);
        }
        Ok(Start::Check)
    }

    /// `grow()` and the push of a new state.
    // ARITH: one more state than the deepest pushed so far.
    #[allow(clippy::arithmetic_side_effects)]
    fn push(&mut self, next_state: i32, change_ord: i32) {
        if self.states.len() == 1 + self.state_upto {
            self.states.push(State::default());
        }
        self.state_upto += 1;
        let st = &mut self.states[self.state_upto];
        st.change_ord = change_ord;
        st.state = next_state;
        st.transition_count = self.accessor.get_num_transitions(next_state);
        self.accessor
            .init_transition(next_state, &mut st.transition);
        st.transition_upto = -1;
        st.transition_max = -1;
    }

    /// `next()`.
    // ARITH: ords stay within `[0, len]` and `state_upto` within the
    // pushed states, as in Java; offsets index `term_bytes`.
    #[allow(clippy::arithmetic_side_effects, clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&'a [u8]> {
        let f = self.field;
        let n = f.len();
        self.on_term = false;
        self.term_ord += 1;
        let mut skip_upto = 0usize;

        if self.term_ord == 0 && f.term_offsets.get(1) == Some(&0) {
            // Special-case empty string.
            if self.runnable.is_accept(self.states[0].state) {
                self.on_term = true;
                return Some(&[]);
            }
            self.term_ord += 1;
        }

        'next_term: loop {
            if self.term_ord >= n {
                return None;
            }
            let ord = self.term_ord as usize;
            if self.term_ord == self.states[self.state_upto].change_ord {
                // Pop.
                self.state_upto = self.state_upto.checked_sub(1)?;
                continue;
            }

            let term_offset = f.offset(ord);
            let term_length = f.offset(ord + 1) - term_offset;
            let skip_offset = f.skip_offsets[ord];
            let num_skips = f.skip_offsets[ord + 1] - skip_offset;

            let label = byte_at(&f.term_bytes, term_offset + self.state_upto);

            let su = self.state_upto;
            while label > self.states[su].transition_max {
                let st = &mut self.states[su];
                st.transition_upto += 1;
                // Java's `==`: `>=` as well once a start term the automaton
                // rejects left `transitionUpto` at the count, where Java
                // reads past the state's transitions.
                if st.transition_upto >= st.transition_count {
                    // We've exhausted transitions leaving this state; force
                    // pop+next/skip now.
                    if su == 0 {
                        self.term_ord = n;
                        return None;
                    }
                    self.term_ord = st.change_ord;
                    skip_upto = 0;
                    self.state_upto -= 1;
                    continue 'next_term;
                }
                self.accessor.get_next_transition(&mut st.transition);
                st.transition_min = st.transition.min;
                st.transition_max = st.transition.max;
            }

            let target_label = self.states[su].transition_min;

            if byte_at(&f.term_bytes, term_offset + su) < target_label {
                let mut low = self.term_ord + 1;
                let mut high = self.states[su].change_ord - 1;
                loop {
                    if low > high {
                        // Label not found.
                        self.term_ord = low;
                        skip_upto = 0;
                        continue 'next_term;
                    }
                    let mut mid = (low + high) >> 1;
                    let at = |m: i32| byte_at(&f.term_bytes, f.offset(m as usize) + su);
                    let cmp = at(mid) - target_label;
                    if cmp < 0 {
                        low = mid + 1;
                    } else if cmp > 0 {
                        high = mid - 1;
                    } else {
                        // Label found; walk backwards to first occurrence.
                        while mid > self.term_ord && at(mid - 1) == target_label {
                            mid -= 1;
                        }
                        self.term_ord = mid;
                        skip_upto = 0;
                        continue 'next_term;
                    }
                }
            }

            let mut next_state = self.runnable.step(self.states[su].state, label);

            if next_state == -1 {
                // Skip.
                if skip_upto < num_skips {
                    self.term_ord = f.skips[skip_offset + skip_upto];
                } else {
                    self.term_ord += 1;
                }
                skip_upto = 0;
            } else if skip_upto < num_skips {
                // Push.
                let change_ord = f.skips[skip_offset + skip_upto];
                skip_upto += 1;
                self.push(next_state, change_ord);
                if self.state_upto == term_length {
                    if self.runnable.is_accept(next_state) {
                        self.on_term = true;
                        return Some(f.term(ord));
                    }
                    self.term_ord += 1;
                    skip_upto = 0;
                }
            } else {
                // Run the non-indexed tail of this term.
                if let Some(suffix) = self.common_suffix {
                    if term_length < suffix.len()
                        || f.term_bytes
                            [term_offset + term_length - suffix.len()..term_offset + term_length]
                            != *suffix
                    {
                        self.term_ord += 1;
                        skip_upto = 0;
                        continue 'next_term;
                    }
                }
                let mut upto = su + 1;
                while upto < term_length {
                    next_state = self
                        .runnable
                        .step(next_state, byte_at(&f.term_bytes, term_offset + upto));
                    if next_state == -1 {
                        self.term_ord += 1;
                        skip_upto = 0;
                        continue 'next_term;
                    }
                    upto += 1;
                }
                if self.runnable.is_accept(next_state) {
                    self.on_term = true;
                    return Some(f.term(ord));
                }
                self.term_ord += 1;
                skip_upto = 0;
            }
        }
    }

    /// The current term's index.
    fn current_ord(&self) -> Option<usize> {
        if !self.on_term {
            return None;
        }
        usize::try_from(self.term_ord).ok()
    }

    /// `term()`: the term `next()` last returned.
    pub fn term(&self) -> Option<&'a [u8]> {
        self.current_ord().map(|o| self.field.term(o))
    }

    /// `ord()`.
    pub fn ord(&self) -> i64 {
        i64::from(self.term_ord)
    }

    /// `docFreq()`, `0` off a term.
    pub fn doc_freq(&self) -> i32 {
        self.current_ord()
            .map_or(0, |o| self.field.terms[o].doc_freq())
    }

    /// `totalTermFreq()`, `0` off a term.
    pub fn total_term_freq(&self) -> i64 {
        self.current_ord()
            .map_or(0, |o| self.field.terms[o].total_term_freq())
    }

    /// `postings(reuse, flags)`; `None` off a term.
    pub fn postings(&self, positions: bool) -> Option<DirectPostingsEnum<'a>> {
        self.current_ord()
            .map(|o| self.field.postings(o, positions))
    }

    /// `seekCeil`/`seekExact(ord)`: unsupported, as in Java.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn seek_ceil(&mut self, _target: &[u8]) -> Result<SeekStatus> {
        Err(Error::Unsupported(
            "seekCeil is not supported on an intersect enum",
        ))
    }
}

/// The `PostingsEnum` a [`DirectField`] term hands out: one variant per
/// Java class.
#[derive(Debug, Clone)]
pub enum DirectPostingsEnum<'a> {
    /// `LowFreqDocsEnumNoTF`.
    LowNoTf(LowFreqDocsEnumNoTf<'a>),
    /// `LowFreqDocsEnumNoPos`.
    LowNoPos(LowFreqDocsEnumNoPos<'a>),
    /// `LowFreqDocsEnum`.
    LowDocs(LowFreqDocsEnum<'a>),
    /// `LowFreqPostingsEnum`.
    LowPostings(LowFreqPostingsEnum<'a>),
    /// `HighFreqDocsEnum`.
    HighDocs(HighFreqDocsEnum<'a>),
    /// `HighFreqPostingsEnum`.
    HighPostings(HighFreqPostingsEnum<'a>),
}

macro_rules! dispatch {
    ($self:expr, $e:ident => $body:expr) => {
        match $self {
            DirectPostingsEnum::LowNoTf($e) => $body,
            DirectPostingsEnum::LowNoPos($e) => $body,
            DirectPostingsEnum::LowDocs($e) => $body,
            DirectPostingsEnum::LowPostings($e) => $body,
            DirectPostingsEnum::HighDocs($e) => $body,
            DirectPostingsEnum::HighPostings($e) => $body,
        }
    };
}

impl<'a> DirectPostingsEnum<'a> {
    /// `nextDoc()`.
    pub fn next_doc(&mut self) -> i32 {
        dispatch!(self, e => e.next_doc())
    }

    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        dispatch!(self, e => e.doc_id())
    }

    /// `freq()`.
    pub fn freq(&self) -> i32 {
        dispatch!(self, e => e.freq())
    }

    /// `nextPosition()`; `-1` without positions.
    // SENTINEL: `-1`, no positions on this field (or not requested).
    pub fn next_position(&mut self) -> i32 {
        match self {
            DirectPostingsEnum::LowPostings(e) => e.next_position(),
            DirectPostingsEnum::HighPostings(e) => e.next_position(),
            // SENTINEL: Java's `nextPosition()` without positions.
            _ => -1,
        }
    }

    /// `startOffset()`; `-1` without offsets.
    pub fn start_offset(&self) -> i32 {
        match self {
            DirectPostingsEnum::LowPostings(e) => e.start_offset,
            DirectPostingsEnum::HighPostings(e) => e.offset_at(1),
            // SENTINEL: Java's `startOffset()` without offsets.
            _ => -1,
        }
    }

    /// `endOffset()`; `-1` without offsets.
    pub fn end_offset(&self) -> i32 {
        match self {
            DirectPostingsEnum::LowPostings(e) => e.end_offset,
            DirectPostingsEnum::HighPostings(e) => e.offset_at(2),
            // SENTINEL: Java's `endOffset()` without offsets.
            _ => -1,
        }
    }

    /// `getPayload()`; `None` for Java's `null`.
    pub fn payload(&self) -> Option<&'a [u8]> {
        match self {
            DirectPostingsEnum::LowPostings(e) => e.payload(),
            DirectPostingsEnum::HighPostings(e) => e.payload(),
            _ => None,
        }
    }

    /// `advance(target)`: `slowAdvance` for the low-frequency enums, a
    /// galloping search for the high-frequency ones.
    pub fn advance(&mut self, target: i32) -> i32 {
        match self {
            DirectPostingsEnum::HighDocs(e) => e.advance(target),
            DirectPostingsEnum::HighPostings(e) => e.advance(target),
            _ => loop {
                let doc = self.next_doc();
                if doc >= target {
                    break doc;
                }
            },
        }
    }

    /// `cost()`.
    pub fn cost(&self) -> i64 {
        dispatch!(self, e => e.cost())
    }
}

/// `postings[i]`, `0` outside it (where Java throws).
fn int_at(ints: &[i32], i: i64) -> i32 {
    usize::try_from(i)
        .ok()
        .and_then(|i| ints.get(i))
        .copied()
        .unwrap_or(0)
}

/// `upto < postings.length`.
fn below(ints: &[i32], upto: i64) -> bool {
    usize::try_from(upto).is_ok_and(|u| u < ints.len())
}

/// `docID()` of the low-frequency enums.
// SENTINEL: `-1` before the first `nextDoc()`, as every `docID()`.
fn low_doc_id(ints: &[i32], upto: i64) -> i32 {
    if upto < 0 {
        // SENTINEL: `docID()` before the first `nextDoc()`.
        -1
    } else if below(ints, upto) {
        int_at(ints, upto)
    } else {
        NO_MORE_DOCS
    }
}

/// `LowFreqDocsEnumNoTF`: `doc` per entry.
#[derive(Debug, Clone)]
pub struct LowFreqDocsEnumNoTf<'a> {
    postings: &'a [i32],
    upto: i64,
}

// ARITH: `upto` walks an in-memory array from `-1`.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> LowFreqDocsEnumNoTf<'a> {
    fn new(postings: &'a [i32]) -> Self {
        LowFreqDocsEnumNoTf { postings, upto: -1 }
    }
    fn next_doc(&mut self) -> i32 {
        self.upto += 1;
        if below(self.postings, self.upto) {
            return int_at(self.postings, self.upto);
        }
        NO_MORE_DOCS
    }
    fn doc_id(&self) -> i32 {
        // SENTINEL-OK: `docID()` hands its `-1` on, as Java's.
        low_doc_id(self.postings, self.upto)
    }
    fn freq(&self) -> i32 {
        1
    }
    fn cost(&self) -> i64 {
        self.postings.len() as i64
    }
}

/// `LowFreqDocsEnumNoPos`: `doc, freq` per entry.
#[derive(Debug, Clone)]
pub struct LowFreqDocsEnumNoPos<'a> {
    postings: &'a [i32],
    upto: i64,
}

// ARITH: `upto` walks an in-memory array from `-2`.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> LowFreqDocsEnumNoPos<'a> {
    fn new(postings: &'a [i32]) -> Self {
        LowFreqDocsEnumNoPos { postings, upto: -2 }
    }
    fn next_doc(&mut self) -> i32 {
        self.upto += 2;
        if below(self.postings, self.upto) {
            return int_at(self.postings, self.upto);
        }
        NO_MORE_DOCS
    }
    fn doc_id(&self) -> i32 {
        // SENTINEL-OK: `docID()` hands its `-1` on, as Java's.
        low_doc_id(self.postings, self.upto)
    }
    fn freq(&self) -> i32 {
        int_at(self.postings, self.upto + 1)
    }
    fn cost(&self) -> i64 {
        (self.postings.len() / 2) as i64
    }
}

/// `LowFreqDocsEnum`: positions present but not asked for, stepped over.
#[derive(Debug, Clone)]
pub struct LowFreqDocsEnum<'a> {
    postings: &'a [i32],
    pos_mult: i64,
    upto: i64,
    freq: i32,
}

// ARITH: `upto` walks an in-memory array from `-2`, by `freq` entries that
// array holds.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> LowFreqDocsEnum<'a> {
    fn new(postings: &'a [i32], pos_mult: i64) -> Self {
        LowFreqDocsEnum {
            postings,
            pos_mult,
            upto: -2,
            freq: 0,
        }
    }
    fn next_doc(&mut self) -> i32 {
        self.upto = self
            .upto
            .saturating_add(2)
            .saturating_add(i64::from(self.freq).saturating_mul(self.pos_mult));
        if below(self.postings, self.upto) {
            self.freq = int_at(self.postings, self.upto + 1);
            return int_at(self.postings, self.upto);
        }
        NO_MORE_DOCS
    }
    fn doc_id(&self) -> i32 {
        // SENTINEL-OK: `docID()` hands its `-1` on, as Java's.
        low_doc_id(self.postings, self.upto)
    }
    fn freq(&self) -> i32 {
        self.freq
    }
    fn cost(&self) -> i64 {
        (self.postings.len() / 2) as i64
    }
}

/// `LowFreqPostingsEnum`: positions, offsets and payloads off the one array.
#[derive(Debug, Clone)]
pub struct LowFreqPostingsEnum<'a> {
    postings: &'a [i32],
    payload_bytes: &'a [u8],
    pos_mult: i64,
    has_offsets: bool,
    has_payloads: bool,
    upto: i64,
    doc_id: i32,
    freq: i32,
    skip_positions: i32,
    start_offset: i32,
    end_offset: i32,
    last_payload_offset: usize,
    payload_offset: usize,
    payload_length: usize,
}

// ARITH: `upto` and the payload offsets walk in-memory arrays, by counts
// those arrays hold.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> LowFreqPostingsEnum<'a> {
    fn new(
        postings: &'a [i32],
        payload_bytes: &'a [u8],
        has_offsets: bool,
        has_payloads: bool,
    ) -> Self {
        let pos_mult = match (has_offsets, has_payloads) {
            (true, true) => 4,
            (true, false) => 3,
            (false, true) => 2,
            (false, false) => 1,
        };
        LowFreqPostingsEnum {
            postings,
            payload_bytes,
            pos_mult,
            has_offsets,
            has_payloads,
            upto: 0,
            doc_id: -1,
            freq: 0,
            skip_positions: 0,
            start_offset: -1,
            end_offset: -1,
            last_payload_offset: 0,
            payload_offset: 0,
            payload_length: 0,
        }
    }

    fn take(&mut self) -> i32 {
        let v = int_at(self.postings, self.upto);
        self.upto += 1;
        v
    }

    fn next_doc(&mut self) -> i32 {
        if self.has_payloads {
            for _ in 0..self.skip_positions {
                self.upto += 1;
                if self.has_offsets {
                    self.upto += 2;
                }
                let len = self.take();
                self.payload_offset = self
                    .payload_offset
                    .saturating_add(usize::try_from(len).unwrap_or(0));
            }
        } else {
            self.upto = self
                .upto
                .saturating_add(self.pos_mult.saturating_mul(i64::from(self.skip_positions)));
        }
        if below(self.postings, self.upto) {
            self.doc_id = self.take();
            self.freq = self.take();
            self.skip_positions = self.freq;
            return self.doc_id;
        }
        self.doc_id = NO_MORE_DOCS;
        NO_MORE_DOCS
    }

    fn doc_id(&self) -> i32 {
        self.doc_id
    }

    fn freq(&self) -> i32 {
        self.freq
    }

    fn next_position(&mut self) -> i32 {
        self.skip_positions -= 1;
        let pos = self.take();
        if self.has_offsets {
            self.start_offset = self.take();
            self.end_offset = self.take();
        }
        if self.has_payloads {
            self.payload_length = usize::try_from(self.take()).unwrap_or(0);
            self.last_payload_offset = self.payload_offset;
            self.payload_offset = self.payload_offset.saturating_add(self.payload_length);
        }
        pos
    }

    fn payload(&self) -> Option<&'a [u8]> {
        if self.payload_length > 0 {
            self.payload_bytes
                .get(self.last_payload_offset..self.last_payload_offset + self.payload_length)
        } else {
            None
        }
    }

    fn cost(&self) -> i64 {
        (self.postings.len() / 2) as i64
    }
}

/// `HighFreqDocsEnum.advance`/`HighFreqPostingsEnum.advance`'s galloping
/// search from `upto + 1`: the new `upto`.
// ARITH: indexes into `doc_ids`, whose length fits an `i32`.
#[allow(clippy::arithmetic_side_effects)]
fn gallop(doc_ids: &[i32], upto: i64, target: i32) -> i64 {
    let len = doc_ids.len() as i64;
    let mut inc = 10i64;
    let mut next_upto = upto + 10;
    let (mut low, mut high);
    loop {
        if next_upto >= len {
            low = next_upto - inc;
            high = len - 1;
            break;
        }
        if target <= int_at(doc_ids, next_upto) {
            low = next_upto - inc;
            high = next_upto;
            break;
        }
        inc *= 2;
        next_upto += inc;
    }
    loop {
        if low > high {
            return low;
        }
        let mid = (low + high) >> 1;
        let cmp = int_at(doc_ids, mid).cmp(&target);
        match cmp {
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid - 1,
            std::cmp::Ordering::Equal => return mid,
        }
    }
}

/// `HighFreqDocsEnum`.
#[derive(Debug, Clone)]
pub struct HighFreqDocsEnum<'a> {
    doc_ids: &'a [i32],
    freqs: Option<&'a [i32]>,
    upto: i64,
    doc_id: i32,
}

// ARITH: `upto` walks `doc_ids` from `-1`.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> HighFreqDocsEnum<'a> {
    fn new(doc_ids: &'a [i32], freqs: Option<&'a [i32]>) -> Self {
        HighFreqDocsEnum {
            doc_ids,
            freqs,
            upto: -1,
            doc_id: -1,
        }
    }
    fn next_doc(&mut self) -> i32 {
        self.upto += 1;
        self.doc_id = if below(self.doc_ids, self.upto) {
            int_at(self.doc_ids, self.upto)
        } else {
            NO_MORE_DOCS
        };
        self.doc_id
    }
    fn doc_id(&self) -> i32 {
        self.doc_id
    }
    fn freq(&self) -> i32 {
        self.freqs.map_or(1, |f| int_at(f, self.upto))
    }
    fn advance(&mut self, target: i32) -> i32 {
        self.upto += 1;
        if self.upto == self.doc_ids.len() as i64 {
            self.doc_id = NO_MORE_DOCS;
            return NO_MORE_DOCS;
        }
        self.upto = gallop(self.doc_ids, self.upto, target);
        self.doc_id = if self.upto == self.doc_ids.len() as i64 {
            NO_MORE_DOCS
        } else {
            int_at(self.doc_ids, self.upto)
        };
        self.doc_id
    }
    fn cost(&self) -> i64 {
        self.doc_ids.len() as i64
    }
}

/// `HighFreqPostingsEnum`.
#[derive(Debug, Clone)]
pub struct HighFreqPostingsEnum<'a> {
    doc_ids: &'a [i32],
    freqs: &'a [i32],
    positions: &'a [Vec<i32>],
    payloads: Option<&'a [Vec<Option<Vec<u8>>>]>,
    pos_jump: i64,
    upto: i64,
    doc_id: i32,
    pos_upto: i64,
}

// ARITH: `upto` walks `doc_ids` from `-1`, `pos_upto` the current
// document's positions from `-pos_jump`.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> HighFreqPostingsEnum<'a> {
    fn new(
        doc_ids: &'a [i32],
        freqs: &'a [i32],
        positions: &'a [Vec<i32>],
        payloads: Option<&'a [Vec<Option<Vec<u8>>>]>,
        has_offsets: bool,
    ) -> Self {
        HighFreqPostingsEnum {
            doc_ids,
            freqs,
            positions,
            payloads,
            pos_jump: if has_offsets { 3 } else { 1 },
            upto: -1,
            doc_id: -1,
            pos_upto: 0,
        }
    }
    fn cur_positions(&self) -> &'a [i32] {
        usize::try_from(self.upto)
            .ok()
            .and_then(|u| self.positions.get(u))
            .map_or(&[], Vec::as_slice)
    }
    fn next_doc(&mut self) -> i32 {
        self.upto += 1;
        if below(self.doc_ids, self.upto) {
            self.pos_upto = -self.pos_jump;
            self.doc_id = int_at(self.doc_ids, self.upto);
        } else {
            self.doc_id = NO_MORE_DOCS;
        }
        self.doc_id
    }
    fn doc_id(&self) -> i32 {
        self.doc_id
    }
    fn freq(&self) -> i32 {
        int_at(self.freqs, self.upto)
    }
    fn next_position(&mut self) -> i32 {
        self.pos_upto += self.pos_jump;
        int_at(self.cur_positions(), self.pos_upto)
    }
    /// `startOffset()` (`which == 1`) or `endOffset()` (`which == 2`).
    // SENTINEL: `-1` without offsets, as Java's.
    fn offset_at(&self, which: i64) -> i32 {
        if self.pos_jump == 3 {
            int_at(self.cur_positions(), self.pos_upto + which)
        } else {
            // SENTINEL: Java's `startOffset()`/`endOffset()` without offsets.
            -1
        }
    }
    fn advance(&mut self, target: i32) -> i32 {
        self.upto += 1;
        if self.upto == self.doc_ids.len() as i64 {
            self.doc_id = NO_MORE_DOCS;
            return NO_MORE_DOCS;
        }
        self.upto = gallop(self.doc_ids, self.upto, target);
        if self.upto == self.doc_ids.len() as i64 {
            self.doc_id = NO_MORE_DOCS;
        } else {
            self.pos_upto = -self.pos_jump;
            self.doc_id = int_at(self.doc_ids, self.upto);
        }
        self.doc_id
    }
    fn payload(&self) -> Option<&'a [u8]> {
        let payloads = self.payloads?;
        let doc = usize::try_from(self.upto).ok()?;
        let pos = usize::try_from(self.pos_upto / self.pos_jump).ok()?;
        payloads.get(doc)?.get(pos)?.as_deref()
    }
    fn cost(&self) -> i64 {
        self.doc_ids.len() as i64
    }
}
