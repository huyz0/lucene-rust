//! Port of `IndexDictionary` and `FSTDictionary`: a field's FST from each
//! block's key to the block's file pointer, and the floor seek that finds
//! the block a term would be in.

use lucene_store::data_output::DataOutput;

use super::{corrupt, length, read_slice, read_vlong, BlockDecoder, BlockEncoder, Result};
use crate::fst::{InputType, PositiveIntOutputs};
use crate::fst_compiler::{FstCompiler, FstCompilerBuilder};
use crate::fst_util::{BytesRefFstEnum, TypedFst};

/// `FSTDictionary`.
#[derive(Debug, Clone)]
pub struct FstDictionary {
    fst: TypedFst<PositiveIntOutputs>,
}

/// `FSTDictionary.Builder`.
pub(crate) struct FstDictionaryBuilder {
    compiler: FstCompiler<PositiveIntOutputs>,
}

impl FstDictionaryBuilder {
    /// `new Builder()`: a default `BYTE1` compiler.
    pub(crate) fn new() -> Self {
        FstDictionaryBuilder {
            compiler: FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>(),
        }
    }

    /// `add(blockKey, blockFilePointer)`.
    ///
    /// # Errors
    /// A key out of order (never, from the block writer).
    pub(crate) fn add(&mut self, block_key: &[u8], block_file_pointer: i64) -> Result<()> {
        Ok(self.compiler.add_bytes(block_key, block_file_pointer)?)
    }

    /// `build()`.
    ///
    /// # Errors
    /// `Corrupted` for a dictionary without a key (a field always has a
    /// block).
    pub(crate) fn build(mut self) -> Result<FstDictionary> {
        let compiled = self
            .compiler
            .compile()
            .ok_or_else(|| corrupt("a field's dictionary has no block"))?;
        Ok(FstDictionary {
            fst: TypedFst::from_compiled(compiled),
        })
    }
}

impl FstDictionary {
    /// `write(output, blockEncoder)`: `fst.save(output, output)`, or its
    /// bytes encoded behind a `vLong` length.
    ///
    /// # Errors
    /// What the encoder fails with.
    pub(crate) fn write(
        &self,
        out: &mut Vec<u8>,
        block_encoder: Option<&dyn BlockEncoder>,
    ) -> Result<()> {
        let bytes = self.fst.save();
        match block_encoder {
            None => out.write_bytes(&bytes),
            Some(encoder) => {
                let encoded = encoder.encode_dictionary(&bytes)?;
                out.write_vlong(encoded.len() as i64);
                out.write_bytes(&encoded);
            }
        }
        Ok(())
    }

    /// `FSTDictionary.read(input, blockDecoder, isFSTOnHeap)` from `buf`,
    /// which starts at the field's `dictionaryStartFP`.
    ///
    /// # Errors
    /// [`super::Error::Fst`] for an FST that does not decode; `Corrupted`
    /// for an encoded length past the end; what the decoder fails with.
    pub(crate) fn read(buf: &[u8], block_decoder: Option<&dyn BlockDecoder>) -> Result<Self> {
        let fst = match block_decoder {
            None => TypedFst::read_prefix(buf)?.0,
            Some(decoder) => {
                let mut pos = 0;
                let num_bytes = length(read_vlong(buf, &mut pos)?)?;
                let decoded = decoder.decode(read_slice(buf, &mut pos, num_bytes)?)?;
                TypedFst::read_prefix(&decoded)?.0
            }
        };
        Ok(FstDictionary { fst })
    }

    /// `browser()`.
    pub fn browser(&self) -> Browser<'_> {
        Browser {
            fst_enum: BytesRefFstEnum::new(&self.fst),
        }
    }

    /// The FST.
    pub fn fst(&self) -> &TypedFst<PositiveIntOutputs> {
        &self.fst
    }
}

/// `FSTDictionary.Browser`: a reusable floor-seeking cursor.
pub struct Browser<'a> {
    fst_enum: BytesRefFstEnum<'a, PositiveIntOutputs>,
}

impl std::fmt::Debug for Browser<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Browser").finish_non_exhaustive()
    }
}

impl Browser<'_> {
    /// `seekBlock(term)`: the file pointer of the block whose key is the
    /// greatest `<= term`, or `-1` when every key is greater.
    ///
    /// # Errors
    /// [`super::Error::Fst`] for a malformed FST.
    // SENTINEL: `-1` = "before the first block"; `BlockReader` and
    // `IntersectBlockReader` map it to the first block or refuse it.
    pub fn seek_block(&mut self, term: &[u8]) -> Result<i64> {
        Ok(self.fst_enum.seek_floor(term)?.map_or(-1, |(_, &fp)| fp))
    }
}
