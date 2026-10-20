//! Port of `FSTOrdsOutputs`: the outputs of the ords block tree's terms
//! index, each the block's frame data plus the range of term ords below the
//! prefix.

use lucene_store::data_output::DataOutput;

use crate::fst_compiler::{FstOutputs, FstReadError, ReverseReader};

/// `FSTOrdsOutputs.Output`: `bytes`, then `startOrd` and `endOrd`, both
/// inclusive; `endOrd` is stored as `Long.MAX_VALUE - lastOrd`, so the
/// FST's `common` (the minimum) of two ranges covers both.
///
/// Java tells `NO_OUTPUT` apart by identity; `newOutput` hands back
/// `NO_OUTPUT` for every `(empty, 0, 0)`, so value equality is the same
/// test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Output {
    /// `bytes`.
    pub bytes: Vec<u8>,
    /// `startOrd`.
    pub start_ord: i64,
    /// `endOrd`: `Long.MAX_VALUE` minus the last ord.
    pub end_ord: i64,
}

impl Output {
    /// `FSTOrdsOutputs.NO_OUTPUT`.
    pub const NO_OUTPUT: Output = Output {
        bytes: Vec::new(),
        start_ord: 0,
        end_ord: 0,
    };

    /// `newOutput(bytes, startOrd, endOrd)`.
    pub fn new(bytes: Vec<u8>, start_ord: i64, end_ord: i64) -> Output {
        Output {
            bytes,
            start_ord,
            end_ord,
        }
    }

    /// The last ord of the range: `Long.MAX_VALUE - endOrd`.
    pub fn last_ord(&self) -> i64 {
        i64::MAX.wrapping_sub(self.end_ord)
    }
}

/// `FSTOrdsOutputs`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FstOrdsOutputs;

impl FstOutputs for FstOrdsOutputs {
    type Value = Output;

    fn no_output() -> Output {
        Output::NO_OUTPUT
    }

    /// The bytes' common prefix and the smaller of each ord.
    fn common(a: &Output, b: &Output) -> Output {
        let n = a
            .bytes
            .iter()
            .zip(&b.bytes)
            .take_while(|(x, y)| x == y)
            .count();
        Output::new(
            a.bytes[..n].to_vec(),
            a.start_ord.min(b.start_ord),
            a.end_ord.min(b.end_ord),
        )
    }

    fn subtract(output: &Output, inc: &Output) -> Output {
        if *inc == Output::NO_OUTPUT {
            return output.clone();
        }
        // Java asserts `output` starts with `inc`.
        let suffix = output.bytes.get(inc.bytes.len()..).unwrap_or_default();
        Output::new(
            suffix.to_vec(),
            output.start_ord.wrapping_sub(inc.start_ord),
            output.end_ord.wrapping_sub(inc.end_ord),
        )
    }

    fn add(prefix: &Output, output: &Output) -> Output {
        let mut sum = prefix.clone();
        Self::add_assign(&mut sum, output);
        sum
    }

    fn add_assign(prefix: &mut Output, output: &Output) {
        if *prefix == Output::NO_OUTPUT {
            prefix.clone_from(output);
        } else if *output != Output::NO_OUTPUT {
            prefix.bytes.extend_from_slice(&output.bytes);
            prefix.start_ord = prefix.start_ord.wrapping_add(output.start_ord);
            prefix.end_ord = prefix.end_ord.wrapping_add(output.end_ord);
        }
    }

    fn write(value: &Output, out: &mut Vec<u8>) {
        // A block's frame data: a few vLongs and floor entries.
        out.write_vint(value.bytes.len() as i32);
        out.write_bytes(&value.bytes);
        out.write_vlong(value.start_ord);
        out.write_vlong(value.end_ord);
    }

    fn read(input: &mut ReverseReader<'_>) -> Result<Output, FstReadError> {
        let len = input.read_vint()?;
        let len = usize::try_from(len)
            .map_err(|_| FstReadError(format!("negative output length {len}")))?;
        let bytes = input.read_bytes_reversed(len)?;
        let start_ord = input.read_vlong()?;
        let end_ord = input.read_vlong()?;
        Ok(Output::new(bytes, start_ord, end_ord))
    }

    fn skip_output(input: &mut ReverseReader<'_>) -> Result<(), FstReadError> {
        let len = input.read_vint()?;
        // Unchecked, as Java's `skipBytes`: the next read fails instead.
        input.skip_bytes(i64::from(len));
        input.read_vlong()?;
        input.read_vlong()?;
        Ok(())
    }
}
