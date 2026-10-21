//! `CorruptingIndexOutput`: an output that, once closed, flips the low bit
//! of one byte of the file it wrote -- after the close, because a flip made
//! while writing would also be in the checksum, which would then match.
//!
//! Differs: Java's is a `FilterIndexOutput` any `Directory` can hand out;
//! here a directory's outputs are one concrete type, so this wraps one for a
//! caller that writes through the [`IndexOutput`] trait, and its
//! [`CorruptingIndexOutput::close`] is explicit (`FsIndexOutput::close`
//! consumes the output; a corruption cannot be returned from a `Drop`).

use lucene_store::{DataOutput, Directory, Error, FsIndexOutput, IndexOutput, Result};

/// `CorruptingIndexOutput(dir, byteToCorrupt, out)`.
pub struct CorruptingIndexOutput<'a> {
    dir: &'a dyn Directory,
    byte_to_corrupt: u64,
    out: FsIndexOutput,
}

impl<'a> CorruptingIndexOutput<'a> {
    /// Wraps `out`, a file of `dir`, to corrupt byte `byte_to_corrupt` of it
    /// once closed.
    pub fn new(dir: &'a dyn Directory, byte_to_corrupt: u64, out: FsIndexOutput) -> Self {
        Self {
            dir,
            byte_to_corrupt,
            out,
        }
    }

    /// `close()`: closes the output, then corrupts the file.
    pub fn close(self) -> Result<()> {
        let name = self.out.name().to_string();
        self.out.close()?;
        corrupt_file(self.dir, &name, self.byte_to_corrupt)
    }
}

/// `CorruptingIndexOutput.corruptFile()`: rewrites `name` through a temp
/// file with the low bit of byte `byte_to_corrupt` flipped.
pub fn corrupt_file(dir: &dyn Directory, name: &str, byte_to_corrupt: u64) -> Result<()> {
    let new_temp_name = {
        let input = dir.open(name)?;
        if byte_to_corrupt >= input.len() as u64 {
            return Err(Error::IllegalArgument(format!(
                "byteToCorrupt={byte_to_corrupt} but file \"{name}\" is only length={}",
                input.len()
            )));
        }
        let mut tmp_out = dir.create_temp_output("tmp", "tmp")?;
        let at = byte_to_corrupt as usize;
        tmp_out.write_bytes(&input[..at]);
        // Flip the 0th bit:
        tmp_out.write_byte(input[at] ^ 1);
        tmp_out.write_bytes(&input[at + 1..]);
        let new_temp_name = tmp_out.name().to_string();
        tmp_out.close()?;
        new_temp_name
    };
    // Delete original and copy corrupt version back:
    dir.delete_file(name)?;
    dir.copy_from(dir, &new_temp_name, name)?;
    dir.delete_file(&new_temp_name)
}

impl DataOutput for CorruptingIndexOutput<'_> {
    fn write_byte(&mut self, b: u8) {
        self.out.write_byte(b);
    }

    /// Byte by byte, as Java's.
    fn write_bytes(&mut self, b: &[u8]) {
        for &byte in b {
            self.write_byte(byte);
        }
    }
}

impl IndexOutput for CorruptingIndexOutput<'_> {
    fn name(&self) -> &str {
        self.out.name()
    }

    fn file_pointer(&self) -> u64 {
        self.out.file_pointer()
    }

    /// The delegate's checksum with its low bit flipped: never the one the
    /// file will have.
    fn checksum(&self) -> u64 {
        self.out.checksum() ^ 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::ByteBuffersDirectory;

    #[test]
    fn flips_the_low_bit_of_one_byte_after_close() {
        let dir = ByteBuffersDirectory::new();
        let mut out = CorruptingIndexOutput::new(&dir, 2, dir.create_output("f").unwrap());
        out.write_bytes(b"abcd");
        out.write_byte(b'e');
        assert_eq!(out.name(), "f");
        assert_eq!(out.file_pointer(), 5);
        let mut plain = dir.create_output("g").unwrap();
        plain.write_bytes(b"abcde");
        assert_eq!(out.checksum(), plain.checksum() ^ 1);
        plain.close().unwrap();
        out.close().unwrap();
        assert_eq!(&*dir.open("f").unwrap(), b"abbde");
        assert_eq!(
            dir.list_all().unwrap(),
            vec!["f".to_string(), "g".to_string()]
        );
    }

    #[test]
    fn a_byte_past_the_end_is_an_illegal_argument() {
        let dir = ByteBuffersDirectory::new();
        let mut out = CorruptingIndexOutput::new(&dir, 3, dir.create_output("f").unwrap());
        out.write_bytes(b"abc");
        let err = out.close().unwrap_err().to_string();
        assert!(
            err.contains("byteToCorrupt=3 but file \"f\" is only length=3"),
            "{err}"
        );
        assert_eq!(&*dir.open("f").unwrap(), b"abc");
    }
}
