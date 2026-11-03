//! The plugin's index-time analysis in Rust: one field of one document,
//! analyzed with a registered analyzer ([`crate::jvm_analysis::register`])
//! and inverted exactly as the plugin's `DocumentEncoder` inverts a field in
//! the JVM -- itself a copy of Lucene's `IndexingChain.PerField.invert`,
//! `TermsHashPerField.add` and `PerField.finish` -- with Java's `int`
//! arithmetic and Java's error messages, since OpenSearch shows those to
//! users.

use lucene_analysis::TokenStream;
use lucene_index::index_writer::{InvertedField, InvertedTerm};
use lucene_util::small_float::int_to_byte4;

use crate::error::{set_last_error, FfiStatus};
use crate::jvm_analysis::Registered;

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
struct TermState {
    term: Vec<u8>,
    hash: u32,
    freq: i32,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    /// From the first occurrence with a payload on; earlier ones empty.
    payloads: Option<Vec<Vec<u8>>>,
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

    /// Room for about `expected` terms before the first rehash.
    fn with_expected(expected: usize) -> TermIds {
        let n = (expected.max(4) * 2).next_power_of_two();
        TermIds {
            slots: vec![Self::EMPTY; n], // alloc-ok: expected is capped at 2^16
        }
    }

    fn hash(term: &[u8]) -> u32 {
        lucene_util::string_helper::murmurhash3_x86_32(term, 0) as u32
    }

    /// The slot holding `term`, or the empty slot where it would go.
    fn find(&self, terms: &[TermState], term: &[u8], hash: u32) -> Result<usize, usize> {
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            match self.slots[i] {
                Self::EMPTY => return Err(i),
                id => {
                    let t = &terms[id as usize];
                    if t.hash == hash && t.term == term {
                        return Ok(id as usize);
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
                let mut i = t.hash as usize & mask;
                while slots[i] != Self::EMPTY {
                    i = (i + 1) & mask;
                }
                slots[i] = id as u32;
            }
            self.slots = slots;
        }
    }
}

impl Inverted<'_> {
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
        let (has_freq, has_prox, has_offsets) =
            (self.has_freq(), self.has_prox(), self.has_offsets());
        if term_freq != 1 && has_prox {
            return Err(illegal(format!(
                "field \"{}\": cannot index positions while using custom TermFrequencyAttribute",
                self.name
            )));
        }
        let hash = TermIds::hash(term);
        let id = match self.ids.find(&self.terms, term, hash) {
            Err(slot) => {
                let id = self.terms.len();
                let freq = if has_freq { term_freq } else { 1 };
                self.max_term_frequency = if has_freq {
                    freq.max(self.max_term_frequency)
                } else {
                    1.max(self.max_term_frequency)
                };
                self.unique_term_count = self.unique_term_count.wrapping_add(1);
                self.terms.push(TermState {
                    term: term.to_vec(),
                    hash,
                    freq,
                    positions: Vec::new(),
                    offsets: Vec::new(),
                    payloads: None,
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
            let position = self.position;
            let t = &mut self.terms[id];
            let earlier = t.positions.len();
            t.positions.push(position);
            if has_offsets {
                t.offsets.push((start, end));
            }
            let payload = payload.filter(|p| !p.is_empty());
            if payload.is_some() && t.payloads.is_none() {
                t.payloads = Some(vec![Vec::new(); earlier]); // alloc-ok: positions already pushed
            }
            if let Some(list) = &mut t.payloads {
                list.push(payload.map(<[u8]>::to_vec).unwrap_or_default());
            }
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
    let mut inv = Inverted {
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
        ids: TermIds::with_expected(expected),
        terms: Vec::with_capacity(expected), // alloc-ok: capped at 2^16
    };
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
    let terms = inv
        .terms
        .into_iter()
        .map(|t| InvertedTerm {
            term: t.term,
            freq: t.freq,
            positions: t.positions,
            offsets: t.offsets,
            payloads: t.payloads.unwrap_or_default(),
        })
        .collect();
    Ok(InvertedField {
        field_number,
        terms,
        norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_analysis::{register, registered, tests::spec};

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
