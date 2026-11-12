//! neural-search's clustered postings: the `.sit` term dictionary and the
//! `.sip` posting clusters of a Seismic segment.
//!
//! `.sit`: index header, then per field `VInt field number`, `VLong term
//! count`, and per term `VInt length`, the term bytes and `VLong` offset of
//! its clusters in `.sip`; then a little-endian `long` (the offset right after
//! the header) and a footer.
//!
//! `.sip`: index header, then per term `VLong cluster count` and per cluster
//! `VLong doc count`, that many `(VInt doc, byte weight)`, a `byte`
//! "should not skip" flag, `VLong summary size` and that many `(VInt folded
//! token, byte weight)`; the same trailing `long` and footer.

use crate::vector::SparseVector;
use crate::{Error, Result};
use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::Input;
use std::collections::HashMap;
use std::sync::Arc;

/// `SparsePostingsConsumer.CODEC_NAME`, in both files' headers.
pub const CODEC_NAME: &str = "SparsePostingsProducer";
/// The only version written so far.
pub const VERSION: i32 = 1;
/// `TERMS_EXTENSION`.
pub const TERMS_EXTENSION: &str = "sit";
/// `POSTING_EXTENSION`.
pub const POSTINGS_EXTENSION: &str = "sip";

/// One cluster of a term's postings.
#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    /// Segment doc ids, ascending.
    pub docs: Vec<i32>,
    /// The term's quantized weight in each doc.
    pub weights: Vec<u8>,
    /// Visit this cluster whatever its summary scores.
    pub should_not_skip: bool,
    /// The cluster's summary vector; `None` when it was written empty.
    pub summary: Option<SparseVector>,
}

/// A segment's clustered postings for every Seismic field in it.
pub struct ClusteredPostings {
    /// Field number to (term bytes to offset in `.sip`).
    fields: HashMap<i32, HashMap<Vec<u8>, u64>>,
    sip: Arc<Input>,
}

fn corrupt(file: &str, what: impl std::fmt::Display) -> Error {
    Error::Corrupted(format!("{file}: {what}"))
}

/// The header both files start with, checked against the segment.
fn check_header(input: &mut SliceInput, segment_id: &[u8; ID_LENGTH]) -> Result<()> {
    codec_util::check_index_header(input, CODEC_NAME, VERSION, VERSION, segment_id, "")?;
    Ok(())
}

/// The `long` before the footer: where the body starts.
fn directory_offset(buf: &[u8], file: &str) -> Result<usize> {
    let end = buf
        .len()
        .checked_sub(codec_util::FOOTER_LENGTH + 8)
        .ok_or_else(|| corrupt(file, "too short"))?;
    let mut input = SliceInput::new(buf);
    input.seek(end)?;
    let offset = input.read_i64()?;
    usize::try_from(offset)
        .ok()
        .filter(|&o| o <= end)
        .ok_or_else(|| corrupt(file, format!("body offset {offset} out of range")))
}

fn vlong_len(input: &mut SliceInput, file: &str, what: &str) -> Result<usize> {
    let v = input.read_vlong()?;
    usize::try_from(v).map_err(|_| corrupt(file, format!("negative {what} {v}")))
}

impl ClusteredPostings {
    /// Reads `.sit` and keeps `.sip` for [`Self::cursor`]. Both files'
    /// headers are checked against `segment_id`; as in neural-search, the
    /// checksums are not verified on open.
    pub fn open(sit: &[u8], sip: Arc<Input>, segment_id: &[u8; ID_LENGTH]) -> Result<Self> {
        let mut input = SliceInput::new(sit);
        check_header(&mut input, segment_id)?;
        let body = directory_offset(sit, TERMS_EXTENSION)?;
        let body_end = sit.len().saturating_sub(codec_util::FOOTER_LENGTH + 8);
        input.seek(body)?;
        let mut fields = HashMap::new();
        // A flushed segment with no field over the threshold has no field
        // count at all, only the trailing long.
        if input.position() < body_end {
            let field_count = input.read_vint()?;
            for _ in 0..field_count {
                let number = input.read_vint()?;
                let term_count = vlong_len(&mut input, TERMS_EXTENSION, "term count")?;
                let mut terms = HashMap::new();
                for _ in 0..term_count {
                    let len = usize::try_from(input.read_vint()?)
                        .map_err(|_| corrupt(TERMS_EXTENSION, "negative term length"))?;
                    let end = input
                        .position()
                        .checked_add(len)
                        .filter(|&e| e <= body_end)
                        .ok_or_else(|| corrupt(TERMS_EXTENSION, "term past the end"))?;
                    let term = input.slice(input.position(), end)?.to_vec();
                    input.seek(end)?;
                    let offset = input.read_vlong()?;
                    let offset = u64::try_from(offset)
                        .map_err(|_| corrupt(TERMS_EXTENSION, "negative postings offset"))?;
                    terms.insert(term, offset);
                }
                fields.insert(number, terms);
            }
        }
        let mut sip_input = SliceInput::new(&sip);
        check_header(&mut sip_input, segment_id)?;
        directory_offset(&sip, POSTINGS_EXTENSION)?;
        Ok(ClusteredPostings { fields, sip })
    }

    /// Whether the segment has clustered postings for `field_number`.
    pub fn has_field(&self, field_number: i32) -> bool {
        self.fields.contains_key(&field_number)
    }

    /// A cursor over the clusters of `term` in `field_number`, read in place
    /// from `.sip` one at a time; `None` when the term is absent or has no
    /// clusters. What a query walks: it decodes the docs of only the clusters
    /// it visits and copies nothing out.
    pub fn cursor(&self, field_number: i32, term: &[u8]) -> Result<Option<ClusterCursor<'_>>> {
        let Some(&offset) = self.fields.get(&field_number).and_then(|t| t.get(term)) else {
            return Ok(None);
        };
        let end = self.sip.len().saturating_sub(codec_util::FOOTER_LENGTH + 8);
        let offset = usize::try_from(offset)
            .ok()
            .filter(|&o| o < end)
            .ok_or_else(|| corrupt(POSTINGS_EXTENSION, "postings offset out of range"))?;
        let mut input = SliceInput::new(&self.sip[..end]);
        input.seek(offset)?;
        let left = vlong_len(&mut input, POSTINGS_EXTENSION, "cluster count")?;
        if left == 0 {
            return Ok(None);
        }
        Ok(Some(ClusterCursor { input, left }))
    }

    /// The clusters of `term` in `field_number`, in stored order; `None`
    /// when the term is absent or has no clusters.
    pub fn clusters(&self, field_number: i32, term: &[u8]) -> Result<Option<Vec<Cluster>>> {
        let Some(mut cursor) = self.cursor(field_number, term)? else {
            return Ok(None);
        };
        let mut clusters = Vec::new();
        while let Some(c) = cursor.next()? {
            let mut docs = Vec::new();
            c.docs_into(&mut docs)?;
            let mut weights = Vec::with_capacity(c.doc_count);
            let mut input = SliceInput::new(c.docs);
            for _ in 0..c.doc_count {
                input.read_vint()?;
                weights.push(input.read_byte()?);
            }
            let summary = if c.summary_len == 0 {
                None
            } else {
                let mut tokens = Vec::with_capacity(c.summary_len);
                let mut ws = Vec::with_capacity(c.summary_len);
                let mut input = SliceInput::new(c.summary);
                for _ in 0..c.summary_len {
                    // `next` checked every summary token fits.
                    tokens.push(u16::try_from(input.read_vint()?).unwrap_or(u16::MAX));
                    ws.push(input.read_byte()?);
                }
                Some(SparseVector::from_sorted(tokens, ws))
            };
            clusters.push(Cluster {
                docs,
                weights,
                should_not_skip: c.should_not_skip,
                summary,
            });
        }
        Ok(Some(clusters))
    }
}

/// The clusters of one term, in stored order, as [`ClusteredPostings::cursor`]
/// reads them: each one checked as [`ClusteredPostings::clusters`] checks it,
/// its docs and summary left in the file until asked for.
pub struct ClusterCursor<'a> {
    input: SliceInput<'a>,
    left: usize,
}

/// One cluster under a [`ClusterCursor`]: its docs and summary as the bytes
/// `.sip` holds them, already checked.
#[derive(Debug, Clone, Copy)]
pub struct ClusterRef<'a> {
    /// Visit this cluster whatever its summary scores.
    pub should_not_skip: bool,
    /// `doc_count` `(VInt doc, byte weight)` pairs.
    docs: &'a [u8],
    doc_count: usize,
    /// `summary_len` `(VInt token, byte weight)` pairs, tokens below 65536.
    summary: &'a [u8],
    summary_len: usize,
}

impl<'a> ClusterCursor<'a> {
    /// The next cluster, `None` after the last.
    ///
    /// # Errors
    /// The cluster runs past the end of the postings, or a summary token is
    /// out of range.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<ClusterRef<'a>>> {
        let Some(left) = self.left.checked_sub(1) else {
            return Ok(None);
        };
        self.left = left;
        let input = &mut self.input;
        let doc_count = vlong_len(input, POSTINGS_EXTENSION, "doc count")?;
        // Each doc takes at least two bytes.
        if doc_count > input.remaining() / 2 {
            return Err(corrupt(POSTINGS_EXTENSION, "doc count past the end"));
        }
        let start = input.position();
        for _ in 0..doc_count {
            input.read_vint()?;
            input.read_byte()?;
        }
        let docs = input.slice(start, input.position())?;
        let should_not_skip = input.read_byte()? != 0;
        let summary_len = vlong_len(input, POSTINGS_EXTENSION, "summary size")?;
        if summary_len > input.remaining() / 2 {
            return Err(corrupt(POSTINGS_EXTENSION, "summary past the end"));
        }
        let start = input.position();
        for _ in 0..summary_len {
            let t = input.read_vint()?;
            if u16::try_from(t).is_err() {
                return Err(corrupt(POSTINGS_EXTENSION, "summary token out of range"));
            }
            input.read_byte()?;
        }
        let summary = input.slice(start, input.position())?;
        Ok(Some(ClusterRef {
            should_not_skip,
            docs,
            doc_count,
            summary,
            summary_len,
        }))
    }
}

impl ClusterRef<'_> {
    /// How many docs the cluster holds.
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Replaces `out` with the cluster's doc ids, ascending.
    ///
    /// # Errors
    /// None in practice: [`ClusterCursor::next`] read these bytes already.
    pub fn docs_into(&self, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        out.reserve(self.doc_count);
        let mut input = SliceInput::new(self.docs);
        for _ in 0..self.doc_count {
            out.push(input.read_vint()?);
            input.read_byte()?;
        }
        Ok(())
    }

    /// `SparseVector.dotProduct` of the summary with the dense query; 0 for
    /// a cluster written without one.
    ///
    /// # Errors
    /// None in practice: [`ClusterCursor::next`] read these bytes already.
    pub fn summary_dot(&self, dense: &[u8]) -> Result<i32> {
        let mut input = SliceInput::new(self.summary);
        let mut score = 0i32;
        for _ in 0..self.summary_len {
            let t = input.read_vint()?;
            let w = input.read_byte()?;
            // Tokens ascend, so the first one past the query ends the scan.
            let Some(&q) = usize::try_from(t).ok().and_then(|t| dense.get(t)) else {
                break;
            };
            // At most 255 * 255; the sum wraps as Java's int does.
            score = score.wrapping_add(i32::from(w).wrapping_mul(i32::from(q)));
        }
        Ok(score)
    }
}

/// One field's clustered postings, as [`write()`] lays them out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldClusters {
    /// The field's number.
    pub number: i32,
    /// Each term's bytes and clusters, in the order they are written.
    pub terms: Vec<(Vec<u8>, Vec<Cluster>)>,
}

/// Writes `value` as a `VLong` of a length or count.
fn write_len(out: &mut Vec<u8>, value: usize) {
    use lucene_store::DataOutput;
    // A Vec length fits in an i64.
    out.write_vlong(i64::try_from(value).unwrap_or(i64::MAX));
}

/// `.sit` and `.sip` for `fields`, in the order given: what
/// `SparsePostingsConsumer` writes for a segment of `segment_id`.
pub fn write(segment_id: &[u8; ID_LENGTH], fields: &[FieldClusters]) -> (Vec<u8>, Vec<u8>) {
    use lucene_store::DataOutput;
    let mut sit = Vec::new();
    let mut sip = Vec::new();
    codec_util::write_index_header(&mut sit, CODEC_NAME, VERSION, segment_id, "");
    codec_util::write_index_header(&mut sip, CODEC_NAME, VERSION, segment_id, "");
    let sit_start = sit.len();
    let sip_start = sip.len();
    sit.write_vint(i32::try_from(fields.len()).unwrap_or(i32::MAX));
    for field in fields {
        sit.write_vint(field.number);
        write_len(&mut sit, field.terms.len());
        for (term, clusters) in &field.terms {
            sit.write_vint(i32::try_from(term.len()).unwrap_or(i32::MAX));
            sit.write_bytes(term);
            write_len(&mut sit, sip.len());
            write_len(&mut sip, clusters.len());
            for c in clusters {
                write_len(&mut sip, c.docs.len());
                for (&d, &w) in c.docs.iter().zip(&c.weights) {
                    sip.write_vint(d);
                    sip.write_byte(w);
                }
                sip.write_byte(u8::from(c.should_not_skip));
                match &c.summary {
                    None => write_len(&mut sip, 0),
                    Some(summary) => {
                        write_len(&mut sip, summary.tokens().len());
                        for (&t, &w) in summary.tokens().iter().zip(summary.weights()) {
                            sip.write_vint(i32::from(t));
                            sip.write_byte(w);
                        }
                    }
                }
            }
        }
    }
    for (out, start) in [(&mut sit, sit_start), (&mut sip, sip_start)] {
        // Header lengths are small.
        out.write_i64(i64::try_from(start).unwrap_or(i64::MAX));
        codec_util::write_footer(out);
    }
    (sit, sip)
}

#[cfg(test)]
impl ClusteredPostings {
    /// Postings for one field built from clusters, laid out as `.sip` is.
    pub(crate) fn from_clusters(field: i32, terms: Vec<(&str, Vec<Cluster>)>) -> Self {
        use lucene_store::DataOutput;
        let mut sip = vec![0u8; 48];
        let mut offsets = HashMap::new();
        for (term, clusters) in terms {
            offsets.insert(term.as_bytes().to_vec(), sip.len() as u64);
            sip.write_vlong(clusters.len() as i64);
            for c in clusters {
                sip.write_vlong(c.docs.len() as i64);
                for (d, w) in c.docs.iter().zip(&c.weights) {
                    sip.write_vint(*d);
                    sip.write_byte(*w);
                }
                sip.write_byte(u8::from(c.should_not_skip));
                let summary = c.summary.unwrap_or_default();
                sip.write_vlong(summary.tokens().len() as i64);
                for (t, w) in summary.tokens().iter().zip(summary.weights()) {
                    sip.write_vint(i32::from(*t));
                    sip.write_byte(*w);
                }
            }
        }
        sip.extend_from_slice(&[0; codec_util::FOOTER_LENGTH + 8]);
        ClusteredPostings {
            fields: HashMap::from([(field, offsets)]),
            sip: Arc::new(Input::external(sip)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::DataOutput;

    const ID: [u8; ID_LENGTH] = [7; ID_LENGTH];

    fn header(out: &mut Vec<u8>) -> i64 {
        codec_util::write_index_header(out, CODEC_NAME, VERSION, &ID, "");
        out.len() as i64
    }

    fn finish(mut out: Vec<u8>, start: i64) -> Vec<u8> {
        out.write_i64(start);
        codec_util::write_footer(&mut out);
        out
    }

    /// `.sit` with one field (number 3) and the given (term, offset) pairs.
    fn sit(terms: &[(&str, i64)]) -> Vec<u8> {
        let mut out = Vec::new();
        let start = header(&mut out);
        out.write_vint(1);
        out.write_vint(3);
        out.write_vlong(terms.len() as i64);
        for (t, o) in terms {
            out.write_vint(t.len() as i32);
            out.write_bytes(t.as_bytes());
            out.write_vlong(*o);
        }
        finish(out, start)
    }

    /// `.sip` whose body is `body`, and the offset the body starts at.
    fn sip(body: impl FnOnce(&mut Vec<u8>)) -> (Arc<Input>, i64) {
        let mut out = Vec::new();
        let start = header(&mut out);
        body(&mut out);
        (Arc::new(Input::external(finish(out, start))), start)
    }

    #[test]
    fn reads_clusters_with_and_without_summaries() {
        let (sip, start) = sip(|o| {
            o.write_vlong(2);
            // Cluster 1: docs 1 and 4, skippable, summary {2: 9}.
            o.write_vlong(2);
            o.write_vint(1);
            o.write_byte(10);
            o.write_vint(4);
            o.write_byte(200);
            o.write_byte(0);
            o.write_vlong(1);
            o.write_vint(2);
            o.write_byte(9);
            // Cluster 2: doc 6, not skippable, no summary.
            o.write_vlong(1);
            o.write_vint(6);
            o.write_byte(3);
            o.write_byte(1);
            o.write_vlong(0);
            // An empty term.
            o.write_vlong(0);
        });
        let empty_at = sip.len() as i64 - codec_util::FOOTER_LENGTH as i64 - 8 - 1;
        let p =
            ClusteredPostings::open(&sit(&[("12", start), ("13", empty_at)]), sip, &ID).unwrap();
        assert!(p.has_field(3));
        assert!(!p.has_field(4));
        let c = p.clusters(3, b"12").unwrap().unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].docs, vec![1, 4]);
        assert_eq!(c[0].weights, vec![10, 200]);
        assert!(!c[0].should_not_skip);
        assert_eq!(c[0].summary.as_ref().unwrap().tokens(), &[2]);
        assert!(c[1].should_not_skip);
        assert_eq!(c[1].summary, None);
        assert_eq!(p.clusters(3, b"13").unwrap(), None);
        assert_eq!(p.clusters(3, b"99").unwrap(), None);
        assert_eq!(p.clusters(4, b"12").unwrap(), None);

        // The cursor reads the same clusters in place.
        let mut cursor = p.cursor(3, b"12").unwrap().unwrap();
        let mut docs = vec![99];
        let first = cursor.next().unwrap().unwrap();
        assert!(!first.should_not_skip);
        assert_eq!(first.doc_count(), 2);
        first.docs_into(&mut docs).unwrap();
        assert_eq!(docs, vec![1, 4]);
        let mut dense = vec![0u8; 3];
        dense[2] = 5;
        assert_eq!(first.summary_dot(&dense).unwrap(), 45);
        // A query shorter than the summary's tokens stops the scan.
        assert_eq!(first.summary_dot(&[1, 1]).unwrap(), 0);
        let second = cursor.next().unwrap().unwrap();
        assert!(second.should_not_skip);
        second.docs_into(&mut docs).unwrap();
        assert_eq!(docs, vec![6]);
        assert_eq!(second.summary_dot(&dense).unwrap(), 0);
        assert!(cursor.next().unwrap().is_none());
        assert!(cursor.next().unwrap().is_none());
        assert!(p.cursor(3, b"13").unwrap().is_none());
        assert!(p.cursor(3, b"99").unwrap().is_none());
        assert!(p.cursor(4, b"12").unwrap().is_none());
    }

    #[test]
    fn a_segment_without_clustered_fields_has_no_field_count() {
        let mut out = Vec::new();
        let start = header(&mut out);
        let sit = finish(out, start);
        let (sip, _) = sip(|_| {});
        let p = ClusteredPostings::open(&sit, sip, &ID).unwrap();
        assert!(!p.has_field(0));
    }

    #[test]
    fn rejects_corrupt_files() {
        let (good_sip, start) = sip(|o| o.write_vlong(0));
        // Wrong segment id.
        assert!(ClusteredPostings::open(&sit(&[]), good_sip.clone(), &[1; ID_LENGTH]).is_err());
        // Too short to hold the trailing long and footer.
        assert!(directory_offset(&[0; 10], "x").is_err());
        // A body offset past the end.
        let mut out = Vec::new();
        header(&mut out);
        let bad = finish(out, 1 << 40);
        assert!(ClusteredPostings::open(&bad, good_sip.clone(), &ID).is_err());
        // A term running past the end of the dictionary.
        let mut out = Vec::new();
        let s = header(&mut out);
        out.write_vint(1);
        out.write_vint(3);
        out.write_vlong(1);
        out.write_vint(1000);
        assert!(ClusteredPostings::open(&finish(out, s), good_sip.clone(), &ID).is_err());
        // A postings offset outside `.sip`.
        let p = ClusteredPostings::open(&sit(&[("1", 1 << 40)]), good_sip, &ID).unwrap();
        assert!(p.clusters(3, b"1").is_err());
        assert!(p.cursor(3, b"1").is_err());
        // Counts larger than what is left.
        for body in [
            |o: &mut Vec<u8>| {
                o.write_vlong(1);
                o.write_vlong(1 << 30);
            },
            |o: &mut Vec<u8>| {
                o.write_vlong(1);
                o.write_vlong(0);
                o.write_byte(0);
                o.write_vlong(1 << 30);
            },
            |o: &mut Vec<u8>| {
                o.write_vlong(1);
                o.write_vlong(0);
                o.write_byte(0);
                o.write_vlong(1);
                o.write_vint(70000);
                o.write_byte(1);
            },
        ] {
            let (sip, start) = sip(body);
            let p = ClusteredPostings::open(&sit(&[("1", start)]), sip, &ID).unwrap();
            assert!(p.clusters(3, b"1").is_err());
            assert!(p.cursor(3, b"1").unwrap().unwrap().next().is_err());
        }
        let _ = start;
    }
}
