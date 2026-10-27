//! Terms dictionaries other than the block tree, over
//! `Lucene104PostingsWriter`'s postings: the `PostingsFormat`s whose
//! `fieldsConsumer` is `new XTermsWriter(state, new
//! Lucene104PostingsWriter(state))` (`BlockTreeOrds`, `UniformSplit`, the
//! test framework's `blockterms` and `Rot13` formats, ...).
//!
//! Such a format writes the same `.doc`/`.pos`/`.pay`/`.psm` as the default
//! one and differs only in the dictionary files; a [`TermsDictionaryFormat`]
//! is that difference. `per_field_postings` writes a field routed to one
//! through its [`TermsDictionaryWriter`], and reads a group of one back by
//! decoding its dictionary ([`TermsDictionaryFormat::read`]) and rewriting
//! it as a block tree over the group's own postings
//! (`postings_writer::write_block_tree_dictionary`).
//!
//! Reading finds the format by the name a field's
//! `PerFieldPostingsFormat.format` attribute records -- Java's
//! `PostingsFormat.forName`, whose SPI registry is [`register`] here.
//! `BlockTreeOrds` is registered from the start; a crate that defines more
//! (the test framework) registers them before reading their segments.

use std::fmt;
use std::sync::RwLock;

use crate::blocktree;
use crate::postings_writer::{
    self, DictionaryField, FieldNorms, FieldPostingsInput, FieldTermStates, PostingsFiles,
    TermSource, WriteOptions,
};
use crate::segment_state::{SegmentReadState, SegmentWriteState};

/// A field's postings with their term type erased, as a format behind
/// `dyn` takes them.
pub type ErasedInput<'a, 'b> = FieldPostingsInput<'a, &'b dyn TermSource>;

/// One `PostingsFormat` that is a terms dictionary over the default
/// postings.
pub trait TermsDictionaryFormat: Send + Sync + fmt::Debug {
    /// `getName()`: what a routed field's `.fnm` records and the stem of its
    /// group's codec suffix.
    fn name(&self) -> &'static str;

    /// The dictionary's file extensions; the first marks a group's presence.
    fn extensions(&self) -> &'static [&'static str];

    /// `fieldsConsumer(state).write(fields, norms)` and `close()` for one
    /// group: its postings, and its dictionary files in
    /// [`Self::extensions`] order. `inputs` are the group's fields with
    /// terms, in name order; `options` decide `.pos`/`.pay` over the whole
    /// segment, as `Lucene104PostingsWriter`'s constructor does. `state`'s
    /// `max_doc` is `0` where the caller does not know it.
    ///
    /// # Errors
    /// What the postings writer or the dictionary fails with.
    fn write(
        &self,
        inputs: &[ErasedInput<'_, '_>],
        norms: &[FieldNorms<'_>],
        options: &WriteOptions,
        state: &SegmentWriteState<'_>,
    ) -> postings_writer::Result<(PostingsFiles, Vec<Vec<u8>>)>;

    /// `fieldsProducer(state)`, enumerated: every field of the group's
    /// dictionary with each term's statistics and `IntBlockTermState`.
    /// `files` are in [`Self::extensions`] order.
    ///
    /// # Errors
    /// What the format's reader rejects.
    fn read(
        &self,
        files: &[&[u8]],
        state: &SegmentReadState<'_>,
    ) -> blocktree::Result<Vec<DictionaryField>>;
}

/// [`TermsDictionaryFormat::write`] for the usual format, whose terms writer
/// takes each field's term states as `Lucene104PostingsWriter` lays out its
/// postings, field by field.
///
/// # Errors
/// What the postings writer or `write_field` fails with.
pub fn write_field_by_field<F>(
    inputs: &[ErasedInput<'_, '_>],
    norms: &[FieldNorms<'_>],
    options: &WriteOptions,
    state: &SegmentWriteState<'_>,
    write_field: F,
) -> postings_writer::Result<PostingsFiles>
where
    F: FnMut(&FieldTermStates<'_>) -> postings_writer::Result<()>,
{
    postings_writer::write_postings(
        inputs,
        norms,
        options.has_prox,
        options.has_payloads_or_offsets,
        state.segment_id,
        state.segment_suffix,
        write_field,
    )
}

static REGISTERED: RwLock<Vec<&'static dyn TermsDictionaryFormat>> = RwLock::new(Vec::new());

/// `PostingsFormat`'s SPI: makes `format` readable by its name. A later
/// registration under a name replaces the earlier one.
pub fn register(format: &'static dyn TermsDictionaryFormat) {
    let mut registered = REGISTERED.write().unwrap_or_else(|e| e.into_inner());
    registered.retain(|f| f.name() != format.name());
    registered.push(format);
}

/// The formats `lucene-codecs` itself defines, under their SPI names.
fn builtin() -> [&'static dyn TermsDictionaryFormat; 3] {
    [
        &crate::blocktreeords::DEFAULT_FORMAT,
        &crate::uniformsplit::DEFAULT_FORMAT,
        &crate::uniformsplit::sharedterms::DEFAULT_FORMAT,
    ]
}

/// `PostingsFormat.forName(name)`, for the dictionaries this module knows.
pub fn lookup(name: &str) -> Option<&'static dyn TermsDictionaryFormat> {
    let builtin = builtin();
    let registered = REGISTERED.read().unwrap_or_else(|e| e.into_inner());
    registered
        .iter()
        .copied()
        .find(|f| f.name() == name)
        .or_else(|| builtin.into_iter().find(|f| f.name() == name))
}

/// Every registered format, the built-in ones included.
pub fn all() -> Vec<&'static dyn TermsDictionaryFormat> {
    let mut out: Vec<&'static dyn TermsDictionaryFormat> = builtin().to_vec();
    let registered = REGISTERED.read().unwrap_or_else(|e| e.into_inner());
    for f in registered.iter() {
        out.retain(|o| o.name() != f.name());
        out.push(*f);
    }
    out
}
