//! `TokenStreamFromTermVector` and `TokenSources`: a stored term vector read
//! back as the token stream the highlighter reads, so a text need not be
//! analyzed again. The vector must carry offsets (and, for the stream to keep
//! Lucene's position increments, positions); a field without offsets falls
//! back to analyzing the stored text.
//!
//! `TermVectorLeafReader` has no counterpart: the extractor indexes whatever
//! stream it is given into its one-document index, and this stream is the
//! vector's own tokens, so the index it builds is the vector's.

use std::collections::BTreeMap;

use lucene_analysis::miscellaneous::LimitTokenOffsetFilter;
use lucene_analysis::{AnalysisError, Analyzer, AttributeSource, TokenStream};
use lucene_codecs::term_vectors::TermVectorField;

use super::{DynStream, Error, Result};

/// One token of the vector, as `TokenLL` holds it.
#[derive(Debug, Clone)]
struct Token {
    term: Vec<u16>,
    position_increment: i32,
    start_offset: i32,
    end_offset_inc: i32,
    payload: Option<Vec<u8>>,
}

/// `Short.MAX_VALUE`: the longest term and offset span a token keeps.
const SHORT_MAX: i32 = i16::MAX as i32;

/// `TokenStreamFromTermVector`.
pub struct TokenStreamFromTermVector {
    atts: AttributeSource,
    vector: TermVectorField,
    max_start_offset: i32,
    include_payloads: bool,
    tokens: Option<Vec<Token>>,
    next: Option<usize>,
}

impl TokenStreamFromTermVector {
    /// `new TokenStreamFromTermVector(vector, maxStartOffset)`: a negative
    /// limit is none.
    ///
    /// # Errors
    /// `IllegalArgumentException` for a vector with neither positions nor
    /// offsets.
    pub fn new(vector: &TermVectorField, max_start_offset: i32) -> Result<Self> {
        if !vector.has_positions && !vector.has_offsets {
            return Err(Error::IllegalArgument(
                "The term vector needs positions and/or offsets.".into(),
            ));
        }
        Ok(TokenStreamFromTermVector {
            atts: AttributeSource::new(),
            vector: vector.clone(),
            max_start_offset: if max_start_offset < 0 {
                i32::MAX
            } else {
                max_start_offset
            },
            include_payloads: false,
            tokens: None,
            next: None,
        })
    }

    /// Also hand out the vector's payloads, as Java does when the consumer
    /// has asked for a `PayloadAttribute` before the stream is read.
    pub fn with_payloads(mut self) -> Self {
        self.include_payloads = true;
        self
    }

    /// `getTermVectorTerms()`.
    pub fn term_vector(&self) -> &TermVectorField {
        &self.vector
    }

    /// `init()`: every occurrence of every term, ordered by position, and
    /// within a position by offsets, with Lucene's position increments.
    fn init(&mut self) -> Vec<Token> {
        let v = &self.vector;
        let with_payloads = self.include_payloads && v.has_payloads;
        let mut by_pos: BTreeMap<i64, Vec<Token>> = BTreeMap::new();
        for t in &v.terms {
            let units: Vec<u16> = String::from_utf8_lossy(&t.term).encode_utf16().collect();
            let len = units
                .len()
                .min(usize::try_from(SHORT_MAX).unwrap_or(usize::MAX));
            let freq = usize::try_from(t.freq).unwrap_or(0);
            for j in 0..freq {
                let mut pos = i64::from(
                    t.positions
                        .as_ref()
                        .and_then(|p| p.get(j))
                        .copied()
                        .unwrap_or(-1),
                );
                let mut token = Token {
                    term: units[..len].to_vec(),
                    position_increment: 0,
                    start_offset: 0,
                    end_offset_inc: 0,
                    payload: None,
                };
                if v.has_offsets {
                    let start = t
                        .start_offsets
                        .as_ref()
                        .and_then(|o| o.get(j))
                        .copied()
                        .unwrap_or(0);
                    let end = t
                        .end_offsets
                        .as_ref()
                        .and_then(|o| o.get(j))
                        .copied()
                        .unwrap_or(0);
                    token.start_offset = start;
                    if token.start_offset > self.max_start_offset {
                        continue;
                    }
                    token.end_offset_inc = (end - start).min(SHORT_MAX);
                    if pos == -1 {
                        pos = i64::from(start >> 3);
                    }
                }
                if with_payloads {
                    token.payload = t
                        .payloads
                        .as_ref()
                        .and_then(|p| p.get(j))
                        .filter(|p| !p.is_empty())
                        .cloned();
                }
                // `insertIntoSortedLinkedList`: before the first token that is
                // not smaller by (start offset, end offset increment)
                let list = by_pos.entry(pos).or_default();
                let at = list.partition_point(|o| {
                    (o.start_offset, o.end_offset_inc) < (token.start_offset, token.end_offset_inc)
                });
                list.insert(at, token);
            }
        }
        let mut out: Vec<Token> = Vec::new();
        let mut prev_token_pos = -1i64;
        for (pos, list) in by_pos {
            let first = out.len();
            out.extend(list);
            if v.has_positions {
                out[first].position_increment =
                    i32::try_from(pos - prev_token_pos).unwrap_or(i32::MAX);
                for t in &mut out[first + 1..] {
                    t.position_increment = 0;
                }
            } else {
                out[first].position_increment = 1;
                for i in first + 1..out.len() {
                    out[i].position_increment =
                        i32::from(out[i - 1].start_offset != out[i].start_offset);
                }
            }
            prev_token_pos = pos;
        }
        out
    }
}

impl TokenStream for TokenStreamFromTermVector {
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }

    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn reset(&mut self) -> std::result::Result<(), AnalysisError> {
        self.next = None;
        Ok(())
    }

    fn increment_token(&mut self) -> std::result::Result<bool, AnalysisError> {
        if self.tokens.is_none() {
            let tokens = self.init();
            self.tokens = Some(tokens);
        }
        let at = self.next.map_or(0, |n| n + 1);
        let Some(token) = self.tokens.as_ref().and_then(|t| t.get(at)).cloned() else {
            return Ok(false);
        };
        self.next = Some(at);
        self.atts.clear_attributes();
        self.atts.set_term_utf16(&token.term);
        self.atts.set_position_increment(token.position_increment)?;
        if self.vector.has_offsets {
            self.atts.set_offset(
                token.start_offset,
                token.start_offset + token.end_offset_inc,
            )?;
        }
        if self.include_payloads && self.vector.has_payloads {
            self.atts.set_payload(token.payload);
        }
        Ok(true)
    }
}

/// `TokenSources`.
pub struct TokenSources;

impl TokenSources {
    /// `getTokenStream(field, tvFields, text, analyzer, maxStartOffset)`: the
    /// field's term vector when it has offsets, else `text` analyzed (cut at
    /// `maxStartOffset` when that is inside the text).
    pub fn get_token_stream<'a>(
        field: &str,
        tv_field: Option<&TermVectorField>,
        text: &str,
        analyzer: &'a Analyzer,
        max_start_offset: i32,
    ) -> Result<DynStream<'a>> {
        if let Some(ts) = Self::get_term_vector_token_stream_or_null(tv_field, max_start_offset)? {
            return Ok(ts);
        }
        let stream = analyzer.token_stream(field, text)?;
        let len = i32::try_from(text.encode_utf16().count()).unwrap_or(i32::MAX);
        if max_start_offset >= 0 && max_start_offset < len - 1 {
            return Ok(DynStream::new(LimitTokenOffsetFilter::new(
                stream,
                max_start_offset,
                true,
            )?));
        }
        Ok(DynStream::new(stream))
    }

    /// `getTermVectorTokenStreamOrNull(field, tvFields, maxStartOffset)`:
    /// `None` without a vector or without offsets in it.
    pub fn get_term_vector_token_stream_or_null(
        tv_field: Option<&TermVectorField>,
        max_start_offset: i32,
    ) -> Result<Option<DynStream<'static>>> {
        match tv_field {
            Some(v) if v.has_offsets => Ok(Some(DynStream::new(TokenStreamFromTermVector::new(
                v,
                max_start_offset,
            )?))),
            _ => Ok(None),
        }
    }

    /// `getTokenStream(Terms)`: the vector as a stream, which must carry
    /// offsets.
    pub fn get_token_stream_from_vector(vector: &TermVectorField) -> Result<DynStream<'static>> {
        if !vector.has_offsets {
            return Err(Error::IllegalArgument(
                "Highlighting requires offsets from the TokenStream.".into(),
            ));
        }
        Ok(DynStream::new(TokenStreamFromTermVector::new(vector, -1)?))
    }

    /// `getTokenStreamWithOffsets(reader, docId, field)` once the vector is
    /// read: `None` without a vector or without offsets in it.
    pub fn get_token_stream_with_offsets(
        vector: Option<&TermVectorField>,
    ) -> Result<Option<DynStream<'static>>> {
        match vector {
            Some(v) if v.has_offsets => Ok(Some(Self::get_token_stream_from_vector(v)?)),
            _ => Ok(None),
        }
    }

    /// `getTokenStream(field, contents, analyzer)`.
    pub fn get_token_stream_from_text<'a>(
        field: &str,
        contents: &str,
        analyzer: &'a Analyzer,
    ) -> Result<DynStream<'a>> {
        Ok(DynStream::new(analyzer.token_stream(field, contents)?))
    }

    /// `getAnyTokenStream(reader, docId, field, analyzer)` once the vector and
    /// the stored text are read: the vector's stream when there is a vector,
    /// else the stored text analyzed.
    ///
    /// # Errors
    /// `IllegalArgumentException` when there is neither a vector nor a stored
    /// text, as `getTokenStream(Document, ...)` throws.
    pub fn get_any_token_stream<'a>(
        vector: Option<&TermVectorField>,
        stored_text: Option<&str>,
        field: &str,
        analyzer: &'a Analyzer,
    ) -> Result<DynStream<'a>> {
        if let Some(v) = vector {
            return Self::get_token_stream_from_vector(v);
        }
        let Some(text) = stored_text else {
            return Err(Error::IllegalArgument(format!(
                "Field {field} in document is not stored and cannot be analyzed"
            )));
        };
        Self::get_token_stream_from_text(field, text, analyzer)
    }
}
