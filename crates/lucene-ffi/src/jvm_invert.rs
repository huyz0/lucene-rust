//! The plugin's index-time analysis in Rust: one field of one document,
//! analyzed with a registered analyzer ([`lucene_engine::analysis::register`])
//! and inverted exactly as the plugin's `DocumentEncoder` inverts a field in
//! the JVM -- itself a copy of Lucene's `IndexingChain.PerField.invert`,
//! `TermsHashPerField.add` and `PerField.finish` -- with Java's `int`
//! arithmetic and Java's error messages, since OpenSearch shows those to
//! users.

use lucene_analysis::TokenStream;
use lucene_index::index_writer::{InvertedField, InvertedTerms};
use lucene_util::small_float::int_to_byte4;

use crate::error::{set_last_error, FfiStatus};
use lucene_engine::analysis::Registered;

/// `IndexWriter.MAX_TERM_LENGTH`.
const MAX_TERM_LENGTH: usize = 32766;
/// `IndexWriter.MAX_POSITION`.
const MAX_POSITION: i32 = i32::MAX - 128;

/// `IndexOptions` by ordinal, as the blob carries it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Options {
    Docs = 1,
    Freqs = 2,
    Positions = 3,
    Offsets = 4,
}

impl Options {
    pub(crate) fn from_ordinal(v: u8) -> Option<Options> {
        Some(match v {
            1 => Options::Docs,
            2 => Options::Freqs,
            3 => Options::Positions,
            4 => Options::Offsets,
            _ => return None,
        })
    }
}

/// How the norm is computed: omitted, or `Similarity.computeNorm` with its
/// `discountOverlaps`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Norm {
    Omit,
    Discount,
    Plain,
}

impl Norm {
    pub(crate) fn from_byte(v: u8) -> Option<Norm> {
        Some(match v {
            0 => Norm::Omit,
            1 => Norm::Discount,
            2 => Norm::Plain,
            _ => return None,
        })
    }
}

fn illegal(message: String) -> FfiStatus {
    set_last_error(message);
    FfiStatus::InvalidArgument
}

/// One term's state within the document (`FreqProxTermsWriterPerField`'s).
/// Its bytes live in [`Inverted::term_bytes`] and its occurrences form a
/// list through [`Inverted::occurrences`], so a new term allocates nothing.
struct TermState {
    /// Where the term's bytes end in `term_bytes`; they start where the
    /// previous term's end.
    end: u32,
    /// [`hash_bytes`] of the term, which the writer reuses.
    hash: u64,
    freq: i32,
    /// The first and last of the term's occurrences, `NONE` before any.
    first: u32,
    last: u32,
    /// Whether any occurrence has a payload; then each one gets one.
    has_payloads: bool,
}

/// One occurrence of a term (for a field that indexes positions).
struct Occurrence {
    position: i32,
    start: i32,
    end: i32,
    /// Where the payload ends in `payload_bytes`; it starts at `payload_from`.
    payload_from: u32,
    payload_end: u32,
    /// The term's next occurrence, `NONE` for the last.
    next: u32,
}

const NONE: u32 = u32::MAX;

use lucene_index::inverter::hash_bytes;

/// The first slot for `hash` in a table of `mask + 1` slots: the hash's top
/// bits. [`hash_bytes`] ends in a multiply, so its low bits see only the
/// term's first bytes.
#[inline]
fn home(hash: u64, mask: usize) -> usize {
    let bits = (mask as u64).count_ones();
    hash.checked_shr(u64::BITS - bits).unwrap_or(0) as usize & mask
}

/// The last inverted field's buffers, emptied: the next field on the same
/// thread grows into them instead of allocating and faulting in new ones.
#[derive(Default)]
struct Spare {
    slots: Vec<u32>,
    terms: Vec<TermState>,
    term_bytes: Vec<u8>,
    occurrences: Vec<Occurrence>,
    payload_bytes: Vec<u8>,
}

/// Above this many occurrences a field's buffers are freed, not kept.
const SPARE_MAX_OCCURRENCES: usize = 1 << 16;

thread_local! {
    static SPARE: std::cell::RefCell<Spare> = std::cell::RefCell::default();
}

/// `FieldInvertState` plus the terms.
struct Inverted<'a> {
    name: &'a str,
    options: Options,
    position: i32,
    length: i32,
    num_overlap: i32,
    offset: i32,
    max_term_frequency: i32,
    unique_term_count: i32,
    last_start_offset: i32,
    last_position: i32,
    ids: TermIds,
    terms: Vec<TermState>,
    term_bytes: Vec<u8>,
    occurrences: Vec<Occurrence>,
    payload_bytes: Vec<u8>,
}

fn term_of<'a>(terms: &[TermState], bytes: &'a [u8], id: usize) -> &'a [u8] {
    let from = id.checked_sub(1).map_or(0, |p| terms[p].end as usize);
    &bytes[from..terms[id].end as usize]
}

/// The document's term ids by term bytes, as `BytesRefHash` keeps them: an
/// open-addressed table of indices into [`Inverted::terms`], so a lookup
/// copies no key and a new term is stored once.
struct TermIds {
    /// `EMPTY`, or an index into the terms.
    slots: Vec<u32>,
}

impl TermIds {
    const EMPTY: u32 = u32::MAX;

    /// Room for about `expected` terms before the first rehash, in `slots`'
    /// allocation when it is large enough.
    fn with_expected(expected: usize, mut slots: Vec<u32>) -> TermIds {
        let n = (expected.max(4) * 2).next_power_of_two();
        slots.clear();
        slots.resize(n, Self::EMPTY); // alloc-ok: expected is capped at 2^16
        TermIds { slots }
    }

    fn hash(term: &[u8]) -> u64 {
        hash_bytes(term)
    }

    /// The slot holding `term`, or the empty slot where it would go.
    fn find(
        &self,
        terms: &[TermState],
        bytes: &[u8],
        term: &[u8],
        hash: u64,
    ) -> Result<usize, usize> {
        let mask = self.slots.len() - 1;
        let mut i = home(hash, mask);
        loop {
            match self.slots[i] {
                Self::EMPTY => return Err(i),
                id => {
                    let id = id as usize;
                    if terms[id].hash == hash && term_of(terms, bytes, id) == term {
                        return Ok(id);
                    }
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Records the newest of `terms` in the empty slot `slot`, growing the
    /// table past half full.
    fn insert(&mut self, slot: usize, terms: &[TermState]) {
        self.slots[slot] = (terms.len() - 1) as u32;
        if terms.len() * 2 > self.slots.len() {
            let mask = self.slots.len() * 2 - 1;
            let mut slots = vec![Self::EMPTY; mask + 1]; // alloc-ok: twice the terms already held
            for (id, t) in terms.iter().enumerate() {
                let mut i = home(t.hash, mask);
                while slots[i] != Self::EMPTY {
                    i = (i + 1) & mask;
                }
                slots[i] = id as u32;
            }
            self.slots = slots;
        }
    }
}

impl<'a> Inverted<'a> {
    fn new(name: &'a str, options: Options, expected: usize) -> Self {
        let spare = SPARE.with(|s| std::mem::take(&mut *s.borrow_mut()));
        Inverted {
            name,
            options,
            position: -1,
            length: 0,
            num_overlap: 0,
            offset: 0,
            max_term_frequency: 0,
            unique_term_count: 0,
            last_start_offset: 0,
            last_position: 0,
            ids: TermIds::with_expected(expected, spare.slots),
            terms: spare.terms,
            term_bytes: spare.term_bytes,
            occurrences: spare.occurrences,
            payload_bytes: spare.payload_bytes,
        }
    }

    /// Gives this field's buffers back for the thread's next one.
    fn recycle(self) {
        let mut spare = Spare {
            slots: self.ids.slots,
            terms: self.terms,
            term_bytes: self.term_bytes,
            occurrences: self.occurrences,
            payload_bytes: self.payload_bytes,
        };
        // a huge field's buffers are not worth holding on to
        if spare.occurrences.capacity() > SPARE_MAX_OCCURRENCES {
            return;
        }
        spare.terms.clear();
        spare.term_bytes.clear();
        spare.occurrences.clear();
        spare.payload_bytes.clear();
        SPARE.with(|s| *s.borrow_mut() = spare);
    }

    /// The terms in first-occurrence order, as the writer takes them.
    fn into_terms(self) -> InvertedTerms {
        let (n, bytes, occurrences) = (
            self.terms.len(),
            self.term_bytes.len(),
            self.occurrences.len(),
        );
        let mut terms = InvertedTerms::with_capacity(n, bytes, occurrences); // alloc-ok: sizes held
        let has_offsets = self.has_offsets();
        for (id, t) in self.terms.iter().enumerate() {
            let mut b = terms.start_hashed(term_of(&self.terms, &self.term_bytes, id), t.hash);
            b.freq(t.freq);
            let mut at = t.first;
            while at != NONE {
                let o = &self.occurrences[at as usize];
                b.position(o.position);
                if has_offsets {
                    b.offsets(o.start, o.end);
                }
                if t.has_payloads {
                    b.payload(&self.payload_bytes[o.payload_from as usize..o.payload_end as usize]);
                }
                at = o.next;
            }
        }
        self.recycle();
        terms
    }

    fn has_freq(&self) -> bool {
        self.options >= Options::Freqs
    }

    fn has_prox(&self) -> bool {
        self.options >= Options::Positions
    }

    fn has_offsets(&self) -> bool {
        self.options >= Options::Offsets
    }

    /// `TermsHashPerField.add`, as `DocumentEncoder.Inverted.add`.
    fn add(
        &mut self,
        term: &[u8],
        term_freq: i32,
        start: i32,
        end: i32,
        payload: Option<&[u8]>,
    ) -> Result<(), FfiStatus> {
        if term.len() > MAX_TERM_LENGTH {
            let prefix: Vec<String> = term[..30].iter().map(|&b| (b as i8).to_string()).collect();
            return Err(illegal(format!(
                "Document contains at least one immense term in field=\"{}\" (whose UTF8 encoding is longer than the max length {MAX_TERM_LENGTH}), all of which were skipped.  Please correct the analyzer to not produce such terms.  The prefix of the first immense term is: '[{}]...'",
                self.name,
                prefix.join(", ")
            )));
        }
        let (has_freq, has_prox) = (self.has_freq(), self.has_prox());
        if term_freq != 1 && has_prox {
            return Err(illegal(format!(
                "field \"{}\": cannot index positions while using custom TermFrequencyAttribute",
                self.name
            )));
        }
        let hash = TermIds::hash(term);
        let id = match self.ids.find(&self.terms, &self.term_bytes, term, hash) {
            Err(slot) => {
                let id = self.terms.len();
                let freq = if has_freq { term_freq } else { 1 };
                self.max_term_frequency = if has_freq {
                    freq.max(self.max_term_frequency)
                } else {
                    1.max(self.max_term_frequency)
                };
                self.unique_term_count = self.unique_term_count.wrapping_add(1);
                self.term_bytes.extend_from_slice(term);
                self.terms.push(TermState {
                    end: self.term_bytes.len() as u32,
                    hash,
                    freq,
                    first: NONE,
                    last: NONE,
                    has_payloads: false,
                });
                self.ids.insert(slot, &self.terms);
                id
            }
            Ok(id) => {
                if !has_freq {
                    if term_freq != 1 {
                        return Err(illegal(format!(
                            "field \"{}\": must index term freq while using custom TermFrequencyAttribute",
                            self.name
                        )));
                    }
                } else {
                    let t = &mut self.terms[id];
                    t.freq = t
                        .freq
                        .checked_add(term_freq)
                        .ok_or_else(|| illegal("integer overflow".to_string()))?;
                    self.max_term_frequency = self.max_term_frequency.max(t.freq);
                }
                id
            }
        };
        if has_prox {
            let payload = payload.unwrap_or_default();
            let payload_from = self.payload_bytes.len() as u32;
            self.payload_bytes.extend_from_slice(payload);
            let at = self.occurrences.len() as u32;
            self.occurrences.push(Occurrence {
                position: self.position,
                start,
                end,
                payload_from,
                payload_end: self.payload_bytes.len() as u32,
                next: NONE,
            });
            let t = &mut self.terms[id];
            t.has_payloads |= !payload.is_empty();
            match t.last {
                NONE => t.first = at,
                last => self.occurrences[last as usize].next = at,
            }
            t.last = at;
        }
        Ok(())
    }
}

/// `PerField.invert` over every instance of a field in one document, then
/// `PerField.finish`'s norm: the field as the writer's
/// [`InvertedField`].
pub(crate) fn invert(
    field_number: i32,
    name: &str,
    registered: &Registered,
    options: Options,
    norm: Norm,
    instances: &[&str],
) -> Result<InvertedField, FfiStatus> {
    let analyzer = registered.analyzer.analyzer();
    let failed = |e: lucene_analysis::AnalysisError| illegal(format!("{e}"));
    // About one distinct term per eight bytes of text; the guess only sizes
    // the buffers.
    let expected = (instances.iter().map(|t| t.len()).sum::<usize>() / 8).min(1 << 16);
    let mut inv = Inverted::new(name, options, expected);
    for text in instances {
        let mut stream = analyzer.token_stream(name, text).map_err(failed)?;
        stream.reset().map_err(failed)?;
        while stream.increment_token().map_err(failed)? {
            let a = stream.attributes();
            let pos_incr = a.position_increment();
            inv.position = inv.position.wrapping_add(pos_incr);
            if inv.position < inv.last_position {
                return Err(illegal(if pos_incr == 0 {
                    format!("first position increment must be > 0 (got 0) for field '{name}'")
                } else if pos_incr < 0 {
                    format!("position increment must be >= 0 (got {pos_incr}) for field '{name}'")
                } else {
                    format!(
                        "position overflowed Integer.MAX_VALUE (got posIncr={pos_incr} lastPosition={} position={}) for field '{name}'",
                        inv.last_position, inv.position
                    )
                }));
            } else if inv.position > MAX_POSITION {
                return Err(illegal(format!(
                    "position {} is too large for field '{name}': max allowed position is {MAX_POSITION}",
                    inv.position
                )));
            }
            inv.last_position = inv.position;
            if pos_incr == 0 {
                inv.num_overlap = inv.num_overlap.wrapping_add(1);
            }
            let start = inv.offset.wrapping_add(a.start_offset());
            let end = inv.offset.wrapping_add(a.end_offset());
            if start < inv.last_start_offset || end < start {
                return Err(illegal(format!(
                    "startOffset must be non-negative, and endOffset must be >= startOffset, and offsets must not go backwards startOffset={start},endOffset={end},lastStartOffset={} for field '{name}'",
                    inv.last_start_offset
                )));
            }
            inv.last_start_offset = start;
            let term_freq = a.term_frequency();
            inv.length = inv
                .length
                .checked_add(term_freq)
                .ok_or_else(|| illegal(format!("too many tokens for field \"{name}\"")))?;
            inv.add(a.term_bytes(), term_freq, start, end, a.payload())?;
        }
        stream.end().map_err(failed)?;
        let a = stream.attributes();
        inv.position = inv.position.wrapping_add(a.position_increment());
        inv.offset = inv.offset.wrapping_add(a.end_offset());
        stream.close().map_err(failed)?;
        // The analyzer's own gaps: the plugin registers the field's analyzer,
        // whose spec carries `getPositionIncrementGap`/`getOffsetGap` for it.
        inv.position = inv
            .position
            .wrapping_add(analyzer.position_increment_gap_for_field(name));
        inv.offset = inv.offset.wrapping_add(analyzer.offset_gap_for_field(name));
    }
    let norm = match norm {
        Norm::Omit => None,
        _ if inv.length == 0 => Some(0),
        n => {
            let num_terms = if options == Options::Docs {
                inv.unique_term_count
            } else if n == Norm::Discount {
                inv.length.wrapping_sub(inv.num_overlap)
            } else {
                inv.length
            };
            let num_terms = u32::try_from(num_terms)
                .map_err(|_| illegal(format!("invalid length: {num_terms}")))?;
            let v = i64::from(int_to_byte4(num_terms) as i8);
            if v == 0 {
                return Err(illegal(format!(
                    "Similarity {} return 0 for non-empty field",
                    registered.similarity
                )));
            }
            Some(v)
        }
    };
    Ok(InvertedField {
        field_number,
        terms: inv.into_terms(),
        norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_engine::analysis::{register, registered, test_support::spec};

    fn reg(stop: bool) -> std::sync::Arc<Registered> {
        let filters: &[(&str, &[(&str, &str)])] = if stop {
            &[("lowercase", &[]), ("stop", &[("words", "w.txt")])]
        } else {
            &[("lowercase", &[])]
        };
        let s = spec(&[], ("standard", &[]), filters, &[("w.txt", "the\n")]);
        let mut blob = Vec::new();
        blob.extend_from_slice(&(s.len() as i32).to_le_bytes());
        blob.extend_from_slice(&s);
        let sim = b"BM25(k1=1.2,b=0.75)";
        blob.extend_from_slice(&(sim.len() as i32).to_le_bytes());
        blob.extend_from_slice(sim);
        let id = register(&blob).unwrap();
        assert_eq!(register(&blob).unwrap(), id, "registered once");
        registered(id as i32).unwrap()
    }

    #[test]
    fn instances_invert_with_gaps_overlaps_and_the_lucene_norm() {
        let r = reg(true);
        // The spec's gaps are 100 positions and 1 offset.
        let f = invert(
            3,
            "body",
            &r,
            Options::Offsets,
            Norm::Discount,
            &["The Fox the fox", "fox"],
        )
        .unwrap();
        assert_eq!(f.field_number, 3);
        let fox = f.terms.iter().find(|t| t.term == b"fox").unwrap();
        // "the" is a stop word: fox at 1 and 3, then 3 + 100 (the gap) + 1.
        assert_eq!(fox.freq, 3);
        assert_eq!(fox.positions, vec![1, 3, 104]);
        assert_eq!(fox.offsets, vec![(4, 7), (12, 15), (16, 19)]);
        // Three tokens: intToByte4(3) = 3.
        assert_eq!(f.norm, Some(3));
        let docs = invert(3, "body", &r, Options::Docs, Norm::Plain, &["a b a"]).unwrap();
        assert_eq!(docs.norm, Some(2), "DOCS counts unique terms");
        assert!(docs
            .terms
            .iter()
            .all(|t| t.freq == 1 && t.positions.is_empty()));
        let empty = invert(3, "body", &r, Options::Positions, Norm::Plain, &["the"]).unwrap();
        assert_eq!(empty.norm, Some(0));
        assert!(empty.terms.is_empty());
        let omit = invert(3, "body", &r, Options::Freqs, Norm::Omit, &["x"]).unwrap();
        assert_eq!(omit.norm, None);
    }

    #[test]
    fn payloads_start_at_the_first_nonempty_one_and_offsets_follow_options() {
        let mut inv = Inverted::new("body", Options::Offsets, 4);
        inv.position = 0;
        inv.add(b"a", 1, 0, 1, None).unwrap();
        inv.add(b"b", 1, 2, 3, None).unwrap();
        inv.position = 1;
        inv.add(b"a", 1, 4, 5, Some(b"xy")).unwrap();
        inv.position = 2;
        inv.add(b"a", 1, 6, 7, Some(b"")).unwrap();
        let terms = inv.into_terms();
        assert_eq!(terms.len(), 2);
        let a = terms.get(0);
        assert_eq!((a.term, a.freq), (&b"a"[..], 3));
        assert_eq!(a.positions, &[0, 1, 2]);
        assert_eq!(a.offsets, &[(0, 1), (4, 5), (6, 7)]);
        let payloads: Vec<&[u8]> = a.payloads.iter().collect();
        assert_eq!(payloads, vec![&b""[..], &b"xy"[..], &b""[..]]);
        let b = terms.get(1);
        assert_eq!((b.term, b.positions), (&b"b"[..], &[0][..]));
        assert!(b.payloads.is_empty());

        let mut inv = Inverted::new("body", Options::Positions, 4);
        inv.position = 0;
        inv.add(b"a", 1, 0, 1, Some(b"p")).unwrap();
        let terms = inv.into_terms();
        assert!(terms.get(0).offsets.is_empty());
        assert_eq!(terms.get(0).payloads.get(0), b"p");
    }

    #[test]
    fn many_terms_keep_first_occurrence_order_through_table_growth() {
        let r = reg(false);
        // The table is sized for one term per eight bytes of text; 3000
        // distinct terms of five bytes each make it grow.
        let words: Vec<String> = (0..3000).map(|i| format!("w{i}")).collect();
        let distinct = words.join(" ");
        let f = invert(
            1,
            "body",
            &r,
            Options::Positions,
            Norm::Plain,
            &[&distinct, "w0 w2999"],
        )
        .unwrap();
        assert_eq!(f.terms.len(), 3000);
        for (i, t) in f.terms.iter().enumerate() {
            assert_eq!(t.term, words[i].as_bytes());
            let again = i == 0 || i == 2999;
            assert_eq!(t.freq, if again { 2 } else { 1 });
            assert_eq!(t.positions[0], i as i32);
        }
    }

    #[test]
    fn an_immense_term_is_java_s_error() {
        let r = reg(false);
        let long = "a".repeat(40000);
        // The standard tokenizer splits at 255 chars, so a keyword-like
        // token this long needs a tokenizer that keeps it whole.
        let s = spec(&[], ("keyword", &[]), &[], &[]);
        let mut blob = Vec::new();
        blob.extend_from_slice(&(s.len() as i32).to_le_bytes());
        blob.extend_from_slice(&s);
        blob.extend_from_slice(&1i32.to_le_bytes());
        blob.push(b'x');
        let kw = registered(register(&blob).unwrap() as i32).unwrap();
        assert_eq!(
            invert(1, "k", &kw, Options::Docs, Norm::Omit, &[&long]).unwrap_err(),
            FfiStatus::InvalidArgument
        );
        let msg = crate::error::last_error();
        assert!(msg.contains("immense term in field=\"k\""), "{msg}");
        assert!(msg.contains("'[97, 97, 97"), "{msg}");
        let _ = r;
    }
}
