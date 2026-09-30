//! Port of `org.apache.lucene.util.packed.PackedDataInput` and
//! `PackedDataOutput`: a bit stream of values of per-call widths, most
//! significant bit first, over a byte-oriented `DataInput`/`DataOutput`.
//!
//! The two are exact mirrors: `write_long(v, bpv)` appends the low `bpv` bits
//! of `v`, high bit first, to the current byte; `flush` pads the last byte
//! with zeros; `read_long(bpv)` reads them back. `skip_to_next_byte` on the
//! reader matches a `flush` on the writer.

use lucene_store::data_input::DataInput;
use lucene_store::data_output::DataOutput;
use lucene_store::Result;

/// `PackedDataInput`.
#[derive(Debug)]
pub struct PackedDataInput<'i, I: DataInput> {
    input: &'i mut I,
    current: u64,
    remaining_bits: u32,
}

impl<'i, I: DataInput> PackedDataInput<'i, I> {
    /// `new PackedDataInput(in)`.
    pub fn new(input: &'i mut I) -> Self {
        PackedDataInput {
            input,
            current: 0,
            remaining_bits: 0,
        }
    }

    /// `readLong(bitsPerValue)`: the next `bits_per_value`-bit value. A width
    /// outside `1..=64` (a Java `assert`) is refused as corruption, since
    /// callers read it from their own metadata.
    pub fn read_long(&mut self, bits_per_value: u32) -> Result<i64> {
        if !(1..=64).contains(&bits_per_value) {
            return Err(lucene_store::Error::Corrupted(format!(
                "PackedDataInput bitsPerValue out of range: {bits_per_value}"
            )));
        }
        let mut left = bits_per_value;
        let mut r: u64 = 0;
        while left > 0 {
            if self.remaining_bits == 0 {
                self.current = self.input.read_byte()? as u64;
                self.remaining_bits = 8;
            }
            let bits = left.min(self.remaining_bits);
            // ARITH: `bits` is in `1..=8` (`min` of a positive `left` and a
            // `remaining_bits` in `1..=8`), so `remaining_bits - bits` and
            // `left - bits` cannot underflow and every shift is below 64. `r`
            // is shifted by at most 8 per step for at most 64 bits in total.
            #[allow(clippy::arithmetic_side_effects)]
            {
                r = (r << bits)
                    | ((self.current >> (self.remaining_bits - bits)) & ((1u64 << bits) - 1));
                left -= bits;
                self.remaining_bits -= bits;
            }
        }
        Ok(r as i64)
    }

    /// `skipToNextByte()`: drop the rest of the current byte.
    pub fn skip_to_next_byte(&mut self) {
        self.remaining_bits = 0;
    }
}

/// `PackedDataOutput`.
#[derive(Debug)]
pub struct PackedDataOutput<'o, O: DataOutput> {
    out: &'o mut O,
    current: u64,
    remaining_bits: u32,
}

impl<'o, O: DataOutput> PackedDataOutput<'o, O> {
    /// `new PackedDataOutput(out)`.
    pub fn new(out: &'o mut O) -> Self {
        PackedDataOutput {
            out,
            current: 0,
            remaining_bits: 8,
        }
    }

    /// `writeLong(value, bitsPerValue)`: append the low `bits_per_value` bits
    /// of `value`. Panics on a width outside `1..=64` (a writer-side bug).
    pub fn write_long(&mut self, value: i64, bits_per_value: u32) {
        assert!(
            (1..=64).contains(&bits_per_value),
            "bitsPerValue={bits_per_value}"
        );
        debug_assert!(
            bits_per_value == 64
                || (value >= 0 && value <= lucene_util::packed::max_value(bits_per_value))
        );
        let value = value as u64;
        let mut left = bits_per_value;
        while left > 0 {
            if self.remaining_bits == 0 {
                self.out.write_byte(self.current as u8);
                self.current = 0;
                self.remaining_bits = 8;
            }
            let bits = self.remaining_bits.min(left);
            // ARITH: as in `read_long`: `bits` is in `1..=8` and at most both
            // `left` and `remaining_bits`, so nothing underflows and every
            // shift is below 64.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.current |= ((value >> (left - bits)) & ((1u64 << bits) - 1))
                    << (self.remaining_bits - bits);
                left -= bits;
                self.remaining_bits -= bits;
            }
        }
    }

    /// `flush()`: write the pending partial byte, if any.
    pub fn flush(&mut self) {
        if self.remaining_bits < 8 {
            self.out.write_byte(self.current as u8);
        }
        self.remaining_bits = 8;
        self.current = 0;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::data_input::SliceInput;

    #[test]
    fn round_trips_mixed_widths_with_flushes() {
        let items: Vec<(i64, u32)> = (1..=64u32)
            .flat_map(|bpv| {
                let max = if bpv == 64 {
                    -1
                } else {
                    lucene_util::packed::max_value(bpv)
                };
                [(max, bpv), (0, bpv), (max / 3, bpv)]
            })
            .collect();
        let mut buf = Vec::new();
        {
            let mut out = PackedDataOutput::new(&mut buf);
            for (i, &(v, b)) in items.iter().enumerate() {
                out.write_long(v, b);
                if i % 7 == 6 {
                    out.flush();
                }
            }
            out.flush();
            out.flush(); // a second flush writes nothing
        }
        let mut input = SliceInput::new(&buf);
        let mut pin = PackedDataInput::new(&mut input);
        for (i, &(v, b)) in items.iter().enumerate() {
            assert_eq!(pin.read_long(b).unwrap(), v, "item {i}");
            if i % 7 == 6 {
                pin.skip_to_next_byte();
            }
        }
        assert!(pin.read_long(1).is_err());
    }

    #[test]
    fn msb_first_layout() {
        let mut buf = Vec::new();
        let mut out = PackedDataOutput::new(&mut buf);
        out.write_long(1, 1);
        out.write_long(0b01, 2);
        out.write_long(0b11111, 5);
        out.write_long(0b101, 3);
        out.flush();
        assert_eq!(buf, vec![0b1011_1111, 0b1010_0000]);
    }

    #[test]
    fn bad_widths() {
        let data = [0u8; 16];
        let mut input = SliceInput::new(&data);
        let mut pin = PackedDataInput::new(&mut input);
        assert!(pin.read_long(0).is_err());
        assert!(pin.read_long(65).is_err());
    }

    #[test]
    #[should_panic(expected = "bitsPerValue")]
    fn writer_rejects_zero_width() {
        let mut buf = Vec::new();
        PackedDataOutput::new(&mut buf).write_long(0, 0);
    }
}
