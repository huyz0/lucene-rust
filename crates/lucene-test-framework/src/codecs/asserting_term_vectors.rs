//! Java's `AssertingTermVectorsFormat` writer state machine and a bounded
//! owned-reader lifecycle seam. Writer assertions remain enabled in release;
//! delegate failures leave assertion state unchanged. Superclass proximity and
//! merge implementations call back through this writer, with no randomness.
//!
//! The reader delegates owned documents without Java's lazy `AssertingFields`,
//! `AssertingTerms`, enum, postings or per-access thread checks. Eager model
//! validation would change their timing and is deliberately absent. The local
//! lifecycle trait supplies clone, merge, close and display operations missing
//! from the cranky reader seam. Its SimpleText adapter shares immutable bytes;
//! clones close independently, while merge handles alias the original close
//! state, matching SimpleText's inherited `getMergeInstance`. Close releases
//! shared data when the final independent group closes, even if closed
//! handles remain alive. Closed get/clone
//! use I/O errors instead of Java runtime exceptions. The optional
//! [`super::asserting_term_vectors_writer::AssertingTermVectorsCodec`] installs
//! this writer state machine over native binary output. Lazy reader assertions,
//! IOContext and whole-codec composition remain absent. The underlying writer's
//! buffered publication and private-buffer accounting limits remain unchanged.

use std::cell::{Ref, RefCell};
use std::fmt;
use std::io;
use std::rc::Rc;

use lucene_codecs::field_infos::FieldInfo;
use lucene_store::DataInput;

use super::cranky_term_vectors::{
    add_prox_default, SimpleTextVectorsFormat, SimpleTextVectorsReader, SimpleTextVectorsWriter,
    TermVectorsFormat, TermVectorsReader, TermVectorsWriter, VectorDocument, VectorState,
};

/// Explicit lifecycle extension; does not add lazy field/term iterator checks.
pub trait OwnedVectorsReader: TermVectorsReader + fmt::Display {
    fn close(&mut self) -> io::Result<()>;
    fn clone_reader(&self) -> io::Result<Self>
    where
        Self: Sized;
    fn get_merge_instance(&self) -> io::Result<Self>
    where
        Self: Sized;
}

pub struct AssertingTermVectorsFormat<F> {
    delegate: F,
}
impl<F> AssertingTermVectorsFormat<F> {
    pub fn new(delegate: F) -> Self {
        Self { delegate }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}
impl<F: TermVectorsFormat> TermVectorsFormat for AssertingTermVectorsFormat<F>
where
    F::Reader: OwnedVectorsReader,
{
    type Writer = AssertingTermVectorsWriter<F::Writer>;
    type Reader = AssertingTermVectorsReader<F::Reader>;
    fn vectors_writer(&mut self, state: &VectorState<'_>) -> io::Result<Self::Writer> {
        Ok(AssertingTermVectorsWriter::new(
            self.delegate.vectors_writer(state)?,
        ))
    }
    fn vectors_reader(&mut self, state: &VectorState<'_>) -> io::Result<Self::Reader> {
        Ok(AssertingTermVectorsReader::new(
            self.delegate.vectors_reader(state)?,
        ))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Undefined,
    Started,
    Finished,
}

pub struct AssertingTermVectorsWriter<W> {
    delegate: W,
    doc_status: Status,
    field_status: Status,
    term_status: Status,
    doc_count: i32,
    field_count: i32,
    term_count: i32,
    position_count: i32,
    has_positions: bool,
}
impl<W> AssertingTermVectorsWriter<W> {
    /// Java: `AssertingTermVectorsWriter` constructor.
    pub fn new(delegate: W) -> Self {
        Self {
            delegate,
            doc_status: Status::Undefined,
            field_status: Status::Undefined,
            term_status: Status::Undefined,
            doc_count: 0,
            field_count: 0,
            term_count: 0,
            position_count: 0,
            has_positions: false,
        }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
    /// Java: `getChildResources`, preserving delegate identity.
    pub fn child_resources(&self) -> impl Iterator<Item = &W> {
        std::iter::once(&self.delegate)
    }
}
impl<W: TermVectorsWriter> TermVectorsWriter for AssertingTermVectorsWriter<W> {
    fn start_document(&mut self, fields: i32) -> io::Result<()> {
        assert_eq!(self.field_count, 0);
        assert_ne!(self.doc_status, Status::Started);
        self.delegate.start_document(fields)?;
        self.doc_status = Status::Started;
        self.field_count = fields;
        self.doc_count = self.doc_count.wrapping_add(1);
        Ok(())
    }
    fn finish_document(&mut self) -> io::Result<()> {
        assert_eq!(self.field_count, 0);
        assert_eq!(self.doc_status, Status::Started);
        self.delegate.finish_document()?;
        self.doc_status = Status::Finished;
        Ok(())
    }
    fn start_field(
        &mut self,
        info: &FieldInfo,
        terms: i32,
        p: bool,
        o: bool,
        y: bool,
    ) -> io::Result<()> {
        assert_eq!(self.term_count, 0);
        assert_eq!(self.doc_status, Status::Started);
        assert_ne!(self.field_status, Status::Started);
        self.delegate.start_field(info, terms, p, o, y)?;
        self.field_status = Status::Started;
        self.term_count = terms;
        self.has_positions = p || o || y;
        Ok(())
    }
    fn finish_field(&mut self) -> io::Result<()> {
        assert_eq!(self.term_count, 0);
        assert_eq!(self.field_status, Status::Started);
        self.delegate.finish_field()?;
        self.field_status = Status::Finished;
        self.field_count = self.field_count.wrapping_sub(1);
        Ok(())
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> io::Result<()> {
        assert_eq!(self.doc_status, Status::Started);
        assert_eq!(self.field_status, Status::Started);
        assert_ne!(self.term_status, Status::Started);
        self.delegate.start_term(term, frequency)?;
        self.term_status = Status::Started;
        self.position_count = if self.has_positions { frequency } else { 0 };
        Ok(())
    }
    fn finish_term(&mut self) -> io::Result<()> {
        assert_eq!(self.position_count, 0);
        assert_eq!(self.doc_status, Status::Started);
        assert_eq!(self.field_status, Status::Started);
        assert_eq!(self.term_status, Status::Started);
        self.delegate.finish_term()?;
        self.term_status = Status::Finished;
        self.term_count = self.term_count.wrapping_sub(1);
        Ok(())
    }
    #[inline]
    fn add_position(&mut self, p: i32, s: i32, e: i32, y: Option<&[u8]>) -> io::Result<()> {
        assert_eq!(self.doc_status, Status::Started);
        assert_eq!(self.field_status, Status::Started);
        assert_eq!(self.term_status, Status::Started);
        self.delegate.add_position(p, s, e, y)?;
        self.position_count = self.position_count.wrapping_sub(1);
        Ok(())
    }
    fn finish(&mut self, docs: i32) -> io::Result<()> {
        assert_eq!(self.doc_count, docs);
        assert_eq!(
            self.doc_status,
            if docs > 0 {
                Status::Finished
            } else {
                Status::Undefined
            }
        );
        assert_ne!(self.field_status, Status::Started);
        assert_ne!(self.term_status, Status::Started);
        self.delegate.finish(docs)
    }
    fn close(&mut self) -> io::Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
    fn ram_bytes_used(&self) -> usize {
        self.delegate.ram_bytes_used()
    }
    // The paired callback baseline was 0.90x Java. Expose the inherited helper
    // and assertion callback together so concrete input calls can specialize.
    // Control flow, assertions, reads and delegated work stay unchanged.
    #[inline]
    fn add_prox(
        &mut self,
        count: i32,
        positions: Option<&mut dyn DataInput>,
        offsets: Option<&mut dyn DataInput>,
    ) -> io::Result<()> {
        add_prox_default(self, count, positions, offsets)
    }
}

pub struct AssertingTermVectorsReader<R> {
    delegate: R,
}
impl<R: OwnedVectorsReader> AssertingTermVectorsReader<R> {
    /// Java evaluates `toString` and checks it is nonnull. Display is nonnull
    /// by type, but still evaluate it to retain the delegate formatting call.
    pub fn new(delegate: R) -> Self {
        let reader = Self { delegate };
        let _ = reader.to_string();
        reader
    }
    pub fn delegate(&self) -> &R {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut R {
        &mut self.delegate
    }
}
impl<R: OwnedVectorsReader> fmt::Display for AssertingTermVectorsReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssertingTermVectorsReader({})", self.delegate)
    }
}
impl<R: OwnedVectorsReader> TermVectorsReader for AssertingTermVectorsReader<R> {
    fn get(&self, doc: i32) -> io::Result<Option<VectorDocument>> {
        self.delegate.get(doc)
    }
    fn check_integrity(&self) -> io::Result<()> {
        self.delegate.check_integrity()
    }
}
impl<R: OwnedVectorsReader> OwnedVectorsReader for AssertingTermVectorsReader<R> {
    fn close(&mut self) -> io::Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
    fn clone_reader(&self) -> io::Result<Self> {
        Ok(Self::new(self.delegate.clone_reader()?))
    }
    fn get_merge_instance(&self) -> io::Result<Self> {
        Ok(Self::new(self.delegate.get_merge_instance()?))
    }
}

/// SimpleText lifecycle over immutable bytes. Clone and merge have different
/// close ownership because Java SimpleText inherits the alias-returning merge.
pub struct OwnedSimpleTextReader {
    reader: Rc<RefCell<Option<Rc<SimpleTextVectorsReader>>>>,
}
impl OwnedSimpleTextReader {
    pub fn open(bytes: Vec<u8>, max_doc: i32) -> io::Result<Self> {
        Ok(Self::new(SimpleTextVectorsReader::open(bytes, max_doc)?))
    }
    fn new(reader: SimpleTextVectorsReader) -> Self {
        Self {
            reader: Rc::new(RefCell::new(Some(Rc::new(reader)))),
        }
    }
    fn reader(&self) -> io::Result<Ref<'_, Rc<SimpleTextVectorsReader>>> {
        Ref::filter_map(self.reader.borrow(), Option::as_ref)
            .map_err(|_| io::Error::other("this TermVectorsReader is closed"))
    }
}
impl fmt::Display for OwnedSimpleTextReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SimpleTextTermVectorsReader")
    }
}
impl TermVectorsReader for OwnedSimpleTextReader {
    fn get(&self, doc: i32) -> io::Result<Option<VectorDocument>> {
        self.reader()?.get(doc)
    }
    fn check_integrity(&self) -> io::Result<()> {
        // Java SimpleTextTermVectorsReader.checkIntegrity is a no-op, even closed.
        Ok(())
    }
}
impl OwnedVectorsReader for OwnedSimpleTextReader {
    fn close(&mut self) -> io::Result<()> {
        // Clear the complete alias group, releasing bytes even when its closed
        // handles survive. Independent clone groups retain their own reference.
        self.reader.borrow_mut().take();
        Ok(())
    }
    fn clone_reader(&self) -> io::Result<Self> {
        let reader = self.reader()?.clone();
        Ok(Self {
            reader: Rc::new(RefCell::new(Some(reader))),
        })
    }
    fn get_merge_instance(&self) -> io::Result<Self> {
        // Java's base method returns this without checking whether it is open.
        Ok(Self {
            reader: self.reader.clone(),
        })
    }
}
pub struct OwnedSimpleTextVectorsFormat;
impl TermVectorsFormat for OwnedSimpleTextVectorsFormat {
    type Writer = SimpleTextVectorsWriter;
    type Reader = OwnedSimpleTextReader;
    fn vectors_writer(&mut self, state: &VectorState<'_>) -> io::Result<Self::Writer> {
        SimpleTextVectorsFormat.vectors_writer(state)
    }
    fn vectors_reader(&mut self, state: &VectorState<'_>) -> io::Result<Self::Reader> {
        Ok(OwnedSimpleTextReader::new(
            SimpleTextVectorsFormat.vectors_reader(state)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closing_last_independent_group_releases_data_with_live_handles() {
        let bytes =
            include_bytes!("../../../../fixtures/testframework/data/asserting_term_vectors.tvc");
        let mut original = OwnedSimpleTextReader::open(bytes.to_vec(), 2).unwrap();
        let weak = Rc::downgrade(original.reader.borrow().as_ref().unwrap());
        let mut independent = original.clone_reader().unwrap();
        let mut alias = original.get_merge_instance().unwrap();
        original.close().unwrap();
        assert!(alias.get(0).is_err());
        assert!(
            weak.upgrade().is_some(),
            "independent clone must retain bytes"
        );
        assert!(independent.get(0).unwrap().is_some());
        independent.close().unwrap();
        assert!(
            weak.upgrade().is_none(),
            "closed handles must not retain bytes"
        );
        assert!(original.get(0).is_err());
        assert!(independent.get(0).is_err());
        assert!(alias.get(0).is_err());
        alias.close().unwrap();
        original.check_integrity().unwrap();
    }
    #[test]
    fn independent_clone_and_alias_merge_close() {
        let bytes =
            include_bytes!("../../../../fixtures/testframework/data/asserting_term_vectors.tvc");
        let mut reader = AssertingTermVectorsReader::new(
            OwnedSimpleTextReader::open(bytes.to_vec(), 2).unwrap(),
        );
        let mut clone = reader.clone_reader().unwrap();
        assert!(clone.get(0).unwrap().is_some());
        clone.close().unwrap();
        assert!(clone.get(0).is_err());
        assert!(reader.get(0).unwrap().is_some());
        let mut merged = reader.get_merge_instance().unwrap();
        merged.close().unwrap();
        assert!(reader.get(0).is_err());
        assert!(reader.clone_reader().is_err());
        assert!(reader.get_merge_instance().unwrap().get(0).is_err());
        reader.check_integrity().unwrap();
        reader.close().unwrap();
        assert_eq!(
            reader.to_string(),
            "AssertingTermVectorsReader(SimpleTextTermVectorsReader)"
        );
        assert!(OwnedSimpleTextReader::open(vec![], 2).is_err());
    }
}
