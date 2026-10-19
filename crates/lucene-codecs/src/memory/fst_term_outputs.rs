//! Port of `org.apache.lucene.codecs.memory.FSTTermOutputs`: the outputs of
//! `FSTTermsWriter`'s FST, one [`TermData`] per term -- its postings
//! metadata bytes (`encodeTerm(absolute=true)`), `docFreq` and
//! `totalTermFreq`.

use lucene_store::data_output::DataOutput;

use crate::fst_compiler::{FstOutputs, FstReadError, ReverseReader};

/// `FSTTermOutputs.TermData`.
///
/// Java tells `NO_OUTPUT` apart by identity; every value this port's FST
/// handles is either [`TermData::NO_OUTPUT`] or a term's own output
/// (`docFreq >= 1`), so value equality is the same test.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TermData {
    /// The postings writer's `encodeTerm` bytes, `None` when it wrote none.
    pub bytes: Option<Vec<u8>>,
    /// `docFreq`, 0 on a shared (non-final) arc.
    pub doc_freq: i32,
    /// `totalTermFreq`, `-1` for a field without frequencies.
    pub total_term_freq: i64,
}

impl TermData {
    /// `FSTTermOutputs.NO_OUTPUT`: `new TermData()`.
    pub const NO_OUTPUT: TermData = TermData {
        bytes: None,
        doc_freq: 0,
        total_term_freq: -1,
    };

    fn is_no_output(&self) -> bool {
        *self == Self::NO_OUTPUT
    }
}

/// `FSTTermOutputs`, for a field whose `IndexOptions` is `DOCS`
/// (`HAS_POS == false`: no `totalTermFreq` on the wire) or anything more
/// (`HAS_POS == true`). Java decides `hasPos` per instance; this port's
/// outputs are static, so the flag is a type parameter.
#[derive(Debug, Clone, Copy, Default)]
pub struct FstTermOutputs<const HAS_POS: bool>;

impl<const HAS_POS: bool> FstOutputs for FstTermOutputs<HAS_POS> {
    type Value = TermData;

    fn no_output() -> TermData {
        TermData::NO_OUTPUT
    }

    /// The output itself when both are equal, else `NO_OUTPUT`: only whole
    /// outputs are shared towards the root.
    fn common(t1: &TermData, t2: &TermData) -> TermData {
        if t1.is_no_output() || t2.is_no_output() || t1 != t2 {
            TermData::NO_OUTPUT
        } else {
            t1.clone()
        }
    }

    fn subtract(t1: &TermData, t2: &TermData) -> TermData {
        if t2.is_no_output() {
            t1.clone()
        } else if t1 == t2 {
            TermData::NO_OUTPUT
        } else {
            t1.clone()
        }
    }

    fn add(t1: &TermData, t2: &TermData) -> TermData {
        if t1.is_no_output() {
            t2.clone()
        } else if t2.is_no_output() || (t2.bytes.is_none() && t2.doc_freq <= 0) {
            t1.clone()
        } else {
            t2.clone()
        }
    }

    // ARITH: `len << 2` is taken only for `len < 32`; `docFreq << 1` and
    // `totalTermFreq - docFreq` are Java's own (wrapping) arithmetic over
    // a term's statistics.
    #[allow(clippy::arithmetic_side_effects)]
    fn write(data: &TermData, out: &mut Vec<u8>) {
        let bytes = data.bytes.as_deref().filter(|b| !b.is_empty());
        let bit0 = i32::from(bytes.is_some());
        let bit1 = i32::from(data.doc_freq != 0) << 1;
        let mut bits = bit0 | bit1;
        if let Some(b) = bytes {
            if b.len() < 32 {
                bits |= (b.len() as i32) << 2;
                out.write_byte(bits as u8);
            } else {
                out.write_byte(bits as u8);
                out.write_vint(b.len() as i32);
            }
            out.write_bytes(b);
        } else {
            out.write_byte(bits as u8);
        }
        if bit1 > 0 {
            if HAS_POS {
                if i64::from(data.doc_freq) == data.total_term_freq {
                    out.write_vint(data.doc_freq.wrapping_shl(1) | 1);
                } else {
                    out.write_vint(data.doc_freq.wrapping_shl(1));
                    out.write_vlong(data.total_term_freq.wrapping_sub(i64::from(data.doc_freq)));
                }
            } else {
                out.write_vint(data.doc_freq);
            }
        }
    }

    fn read(input: &mut ReverseReader<'_>) -> Result<TermData, FstReadError> {
        let bits = input.read_byte()?;
        let (has_bytes, has_stats) = (bits & 1 != 0, bits & 2 != 0);
        let mut bytes = None;
        if has_bytes {
            let len = bytes_len(bits, input)?;
            bytes = Some(input.read_bytes_reversed(len)?);
        }
        let mut doc_freq = 0;
        let mut total_term_freq = -1;
        if has_stats {
            let code = input.read_vint()?;
            if HAS_POS {
                doc_freq = ((code as u32) >> 1) as i32;
                total_term_freq = i64::from(doc_freq);
                if code & 1 == 0 {
                    total_term_freq = total_term_freq.wrapping_add(input.read_vlong()?);
                }
            } else {
                doc_freq = code;
            }
        }
        Ok(TermData {
            bytes,
            doc_freq,
            total_term_freq,
        })
    }

    fn skip_output(input: &mut ReverseReader<'_>) -> Result<(), FstReadError> {
        let bits = input.read_byte()?;
        if bits & 1 != 0 {
            let len = bytes_len(bits, input)?;
            // Unchecked, as Java's `skipBytes`: the next read fails instead.
            input.skip_bytes(len as i64);
        }
        if bits & 2 != 0 {
            let code = input.read_vint()?;
            if HAS_POS && code & 1 == 0 {
                input.read_vlong()?;
            }
        }
        Ok(())
    }
}

/// The metadata length: the flags byte's upper six bits, or a vint after it
/// when they are 0.
fn bytes_len(bits: u8, input: &mut ReverseReader<'_>) -> Result<usize, FstReadError> {
    let inline = usize::from(bits >> 2);
    if inline != 0 {
        return Ok(inline);
    }
    let len = input.read_vint()?;
    usize::try_from(len).map_err(|_| FstReadError(format!("negative metadata length {len}")))
}
