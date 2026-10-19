//! `MultiPassIndexSplitter`: splits an index into several equal parts, by
//! sequential document-id ranges or round-robin. Each part is one
//! `addIndexes(CodecReader...)` over the input with every document outside
//! the part hidden, so the source is read once per part and is never changed.

use lucene_store::directory::Directory;

use super::{index_error, live_flags, open_writer};
use crate::directory_reader::DirectoryReader;
use crate::{Error, Result};

/// `MultiPassIndexSplitter`.
#[derive(Debug, Default, Clone, Copy)]
pub struct MultiPassIndexSplitter;

impl MultiPassIndexSplitter {
    /// `split(in, outputs, seq)`: `inputs` are the indexes `in` is made of
    /// (a `MultiReader` over them when there are several; their document ids
    /// run on from one to the next). `seq` splits into equal increasing id
    /// ranges, the last part taking the rounding remainder; otherwise
    /// document ids go round-robin. Documents the input had deleted are in no
    /// part.
    ///
    /// # Errors
    /// `IOException`s of Java as [`Error::IllegalArgument`]: fewer than two
    /// outputs, or fewer than two live documents.
    pub fn split(
        &self,
        inputs: &[&dyn Directory],
        outputs: &[&dyn Directory],
        seq: bool,
    ) -> Result<()> {
        if outputs.len() < 2 {
            return Err(Error::IllegalArgument("Invalid number of outputs.".into()));
        }
        let readers = inputs
            .iter()
            .map(|d| DirectoryReader::open(*d).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        // Per source, per segment, whether each document is live; a
        // document's id is its place counting every source in turn.
        let live: Vec<Vec<Vec<bool>>> = readers.iter().map(live_flags).collect();
        let num_docs: usize = live.iter().flatten().flatten().filter(|l| **l).count();
        if num_docs < 2 {
            return Err(Error::IllegalArgument(
                "Not enough documents for splitting".into(),
            ));
        }
        let max_doc: usize = live.iter().flatten().map(Vec::len).sum();
        let num_parts = outputs.len();
        let part_len = max_doc / num_parts;
        for (i, output) in outputs.iter().enumerate() {
            // The first global id of each segment, in source order.
            let mut bases = Vec::new();
            let mut base = 0usize;
            for source in &live {
                bases.push(
                    source
                        .iter()
                        .map(|s| {
                            let b = base;
                            base += s.len();
                            b
                        })
                        .collect::<Vec<_>>(),
                );
            }
            let in_part = |global: usize| {
                if seq {
                    let lo = part_len * i;
                    let hi = lo + part_len;
                    global >= lo && (i == num_parts - 1 || global < hi)
                } else {
                    (global + num_parts - i).is_multiple_of(num_parts)
                }
            };
            let mut w = open_writer(*output)?;
            w.add_indexes_filtered(inputs, &|source, segment, doc| {
                let d = usize::try_from(doc).unwrap_or(0);
                live[source][segment][d] && in_part(bases[source][segment] + d)
            })
            .map_err(index_error)?;
            w.close().map_err(index_error)?;
        }
        Ok(())
    }
}
