//! Java's `AssertingStoredFieldsFormat`, with the writer's three-state document
//! protocol and a reader lifecycle seam. Assertions remain enabled in release.
//! Reader document calls check the acquisition thread before document bounds;
//! cloning a merge instance is rejected. Each successful close delegates twice.
//!
//! The seven Java writeField overloads use the existing six-kind FieldValue
//! plus a separate DataInput callback. Nullable delegates and visitor objects
//! are unrepresentable. Reader Display strings replace Java toString; construction
//! evaluates the string as Java's nonnull assertion does. Infallible clone and
//! merge acquisition preserve Java's callback shape (closed native acquisition
//! panics, while document and integrity operations return typed I/O errors).
//!
//! The Lucene90 adapter reuses the production reader and the established
//! cranky_stored_fields writer adapter without injecting failures. Borrowed file
//! bytes and shared parsed metadata replace owned IndexInputs. The document bound
//! comes from parsed metadata rather than SegmentInfo. The concrete Display uses
//! `Lucene90StoredFieldsReader(maxDoc=N)` rather than Java's compression-mode and
//! chunk-size diagnostic string. Closing one handle does not invalidate
//! independently cloned handles, matching ByteBuffers inputs.
//! This does not model backends that invalidate clones on original close. Merge
//! handles use the ordinary native range decoder, without Java's merge-only
//! decompressed-chunk cache. The writer retains the existing buffered document,
//! output/error timing and accounting limits; IOContext, SegmentInfo routing and
//! Codec/IndexWriter installation remain outside this explicit format seam.

use std::fmt;
use std::thread::{self, ThreadId};

use lucene_codecs::stored_fields::{self, FieldValue, StoredFieldVisitor};
use lucene_store::DataInput;

use super::cranky_stored_fields::{self, StoredFieldsFormat as ByteFormat};
pub use super::cranky_stored_fields::{
    ChildResource, StoredFieldsReadState, StoredFieldsSource, StoredFieldsState, StoredFieldsWriter,
};

/// Reader lifecycle callbacks absent from the byte-only production reader.
pub trait StoredFieldsReader: StoredFieldsSource + fmt::Display {
    fn max_doc(&self) -> i32;
    fn close(&mut self) -> stored_fields::Result<()>;
    fn clone_reader(&self) -> Self
    where
        Self: Sized;
    fn get_merge_instance(&self) -> Self
    where
        Self: Sized;
}

/// The existing writer seam plus reader lifecycle callbacks.
pub trait StoredFieldsFormat {
    type Writer<'a>: StoredFieldsWriter;
    type Reader<'a>: StoredFieldsReader;
    fn fields_writer<'a>(
        &mut self,
        state: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>>;
    fn fields_reader<'a>(
        &mut self,
        state: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>>;
}

pub struct AssertingStoredFieldsFormat<F> {
    delegate: F,
}
impl<F> AssertingStoredFieldsFormat<F> {
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
impl<F: StoredFieldsFormat> StoredFieldsFormat for AssertingStoredFieldsFormat<F> {
    type Writer<'a> = AssertingStoredFieldsWriter<F::Writer<'a>>;
    type Reader<'a> = AssertingStoredFieldsReader<F::Reader<'a>>;
    /// Java: `AssertingStoredFieldsFormat.fieldsWriter`.
    fn fields_writer<'a>(
        &mut self,
        state: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        Ok(AssertingStoredFieldsWriter::new(
            self.delegate.fields_writer(state)?,
        ))
    }
    /// Java: `AssertingStoredFieldsFormat.fieldsReader`.
    fn fields_reader<'a>(
        &mut self,
        state: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        let reader = self.delegate.fields_reader(state)?;
        let max_doc = reader.max_doc();
        Ok(AssertingStoredFieldsReader::new(reader, max_doc))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Undefined,
    Started,
    Finished,
}
pub struct AssertingStoredFieldsWriter<W> {
    delegate: W,
    num_written: i32,
    doc_status: Status,
}
impl<W> AssertingStoredFieldsWriter<W> {
    /// Java: `AssertingStoredFieldsWriter(in)`.
    pub fn new(delegate: W) -> Self {
        Self {
            delegate,
            num_written: 0,
            doc_status: Status::Undefined,
        }
    }
    pub fn delegate(&self) -> &W {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut W {
        &mut self.delegate
    }
}
impl<W: StoredFieldsWriter> StoredFieldsWriter for AssertingStoredFieldsWriter<W> {
    /// Java: `AssertingStoredFieldsWriter.startDocument`.
    fn start_document(&mut self) -> stored_fields::Result<()> {
        assert_ne!(self.doc_status, Status::Started);
        self.delegate.start_document()?;
        self.num_written = self.num_written.wrapping_add(1);
        self.doc_status = Status::Started;
        Ok(())
    }
    /// Java: `AssertingStoredFieldsWriter.finishDocument`.
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        assert_eq!(self.doc_status, Status::Started);
        self.delegate.finish_document()?;
        self.doc_status = Status::Finished;
        Ok(())
    }
    /// Java: the six scalar/BytesRef/String `writeField` overloads.
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        assert_eq!(self.doc_status, Status::Started);
        self.delegate.write_field(number, value)
    }
    /// Java: `writeField(FieldInfo,StoredFieldDataInput)`.
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        assert_eq!(self.doc_status, Status::Started);
        self.delegate.write_field_data_input(number, input, length)
    }
    /// Java: `AssertingStoredFieldsWriter.finish`; count assertion is after delegation.
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        assert_eq!(
            self.doc_status,
            if num_docs > 0 {
                Status::Finished
            } else {
                Status::Undefined
            }
        );
        self.delegate.finish(num_docs)?;
        assert_eq!(num_docs, self.num_written);
        Ok(())
    }
    /// Java: `AssertingStoredFieldsWriter.close`.
    fn close(&mut self) -> stored_fields::Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
    fn ram_bytes_used(&self) -> usize {
        self.delegate.ram_bytes_used()
    }
    fn child_resources(&self) -> &[ChildResource] {
        self.delegate.child_resources()
    }
    // Java inherits StoredFieldsWriter.merge: the shared trait default re-enters
    // this wrapper for every document/field and for its final finish assertion.
}

pub struct AssertingStoredFieldsReader<R> {
    delegate: R,
    max_doc: i32,
    merging: bool,
    creation_thread: ThreadId,
}
impl<R: StoredFieldsReader> AssertingStoredFieldsReader<R> {
    /// Java: `AssertingStoredFieldsReader(in,maxDoc,false)`.
    pub fn new(delegate: R, max_doc: i32) -> Self {
        Self::with_merging(delegate, max_doc, false)
    }
    /// Java constructor with its explicit merge flag; useful for recording delegates.
    pub fn with_merging(delegate: R, max_doc: i32, merging: bool) -> Self {
        let reader = Self {
            delegate,
            max_doc,
            merging,
            creation_thread: thread::current().id(),
        };
        // Java's constructor evaluates toString even though Rust strings cannot be null.
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
impl<R: StoredFieldsReader> StoredFieldsSource for AssertingStoredFieldsReader<R> {
    /// Java: `AssertingStoredFieldsReader.checkIntegrity`.
    fn check_integrity(&self) -> stored_fields::Result<()> {
        self.delegate.check_integrity()
    }
    /// Java: `AssertingStoredFieldsReader.document`, including ordinary-reader thread checks.
    fn visit_document(
        &self,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> stored_fields::Result<()> {
        assert_eq!(self.creation_thread, thread::current().id(), "StoredFieldsReader are only supposed to be consumed in the thread in which they have been acquired.");
        assert!(doc >= 0 && doc < self.max_doc);
        self.delegate.visit_document(doc, visitor)
    }
}
impl<R: StoredFieldsReader> StoredFieldsReader for AssertingStoredFieldsReader<R> {
    fn max_doc(&self) -> i32 {
        self.max_doc
    }
    /// Java: `AssertingStoredFieldsReader.close`.
    fn close(&mut self) -> stored_fields::Result<()> {
        self.delegate.close()?;
        self.delegate.close()
    }
    /// Java: `AssertingStoredFieldsReader.clone`.
    fn clone_reader(&self) -> Self {
        assert!(!self.merging, "Merge instances do not support cloning");
        Self::with_merging(self.delegate.clone_reader(), self.max_doc, false)
    }
    /// Java: `AssertingStoredFieldsReader.getMergeInstance`.
    fn get_merge_instance(&self) -> Self {
        Self::with_merging(self.delegate.get_merge_instance(), self.max_doc, true)
    }
}
impl<R: fmt::Display> fmt::Display for AssertingStoredFieldsReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssertingStoredFieldsReader({})", self.delegate)
    }
}

/// Lucene90 bytes and established writer lifecycle, without random injection.
pub struct Lucene90StoredFieldsFormat;
pub struct Lucene90StoredFieldsReader<'a> {
    reader: Option<stored_fields::StoredFieldsReader<'a>>,
    fdt: &'a [u8],
    fdx: &'a [u8],
    max_doc: i32,
}
fn closed() -> stored_fields::Error {
    lucene_store::Error::Io(std::io::Error::other("closed stored fields reader")).into()
}
impl StoredFieldsFormat for Lucene90StoredFieldsFormat {
    type Writer<'a> = cranky_stored_fields::Lucene90StoredFieldsWriter<'a>;
    type Reader<'a> = Lucene90StoredFieldsReader<'a>;
    fn fields_writer<'a>(
        &mut self,
        state: StoredFieldsState<'a>,
    ) -> stored_fields::Result<Self::Writer<'a>> {
        cranky_stored_fields::Lucene90StoredFieldsFormat.fields_writer(state)
    }
    fn fields_reader<'a>(
        &mut self,
        state: StoredFieldsReadState<'a>,
    ) -> stored_fields::Result<Self::Reader<'a>> {
        let reader = cranky_stored_fields::Lucene90StoredFieldsFormat.fields_reader(state)?;
        let max_doc = reader.max_doc();
        Ok(Lucene90StoredFieldsReader {
            reader: Some(reader),
            fdt: state.fdt,
            fdx: state.fdx,
            max_doc,
        })
    }
}
impl StoredFieldsSource for Lucene90StoredFieldsReader<'_> {
    fn check_integrity(&self) -> stored_fields::Result<()> {
        self.reader.as_ref().ok_or_else(closed)?.check_integrity()
    }
    fn visit_document(
        &self,
        doc: i32,
        visitor: &mut dyn StoredFieldVisitor,
    ) -> stored_fields::Result<()> {
        self.reader
            .as_ref()
            .ok_or_else(closed)?
            .visit_document(doc, visitor)
    }
}
impl StoredFieldsReader for Lucene90StoredFieldsReader<'_> {
    fn max_doc(&self) -> i32 {
        self.max_doc
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.reader = None;
        Ok(())
    }
    fn clone_reader(&self) -> Self {
        Self {
            reader: Some(
                self.reader
                    .as_ref()
                    .expect("closed stored fields reader")
                    .rebind(self.fdt, self.fdx),
            ),
            fdt: self.fdt,
            fdx: self.fdx,
            max_doc: self.max_doc,
        }
    }
    fn get_merge_instance(&self) -> Self {
        self.clone_reader()
    }
}
impl fmt::Display for Lucene90StoredFieldsReader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lucene90StoredFieldsReader(maxDoc={})", self.max_doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::stored_fields::Mode;
    use lucene_store::ByteBuffersDirectory;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn document_counter_uses_java_wrapping_after_successful_start() {
        let directory = ByteBuffersDirectory::new();
        let delegate = Lucene90StoredFieldsFormat
            .fields_writer(StoredFieldsState {
                directory: &directory,
                segment: "_wrap",
                id: &[0; 16],
                suffix: "",
                mode: Mode::BestSpeed,
            })
            .unwrap();
        let mut writer = AssertingStoredFieldsWriter::new(delegate);
        writer.num_written = i32::MAX;
        writer.start_document().unwrap();
        assert_eq!(writer.num_written, i32::MIN);
        assert_eq!(writer.doc_status, Status::Started);
        writer.close().unwrap();
    }

    #[test]
    fn native_closed_handle_preserves_document_and_clone_error_boundaries() {
        let fdt = include_bytes!(
            "../../../../fixtures/testframework/data/asserting_stored_fields_speed.fdt"
        );
        let fdx = include_bytes!(
            "../../../../fixtures/testframework/data/asserting_stored_fields_speed.fdx"
        );
        let fdm = include_bytes!(
            "../../../../fixtures/testframework/data/asserting_stored_fields_speed.fdm"
        );
        let id = std::array::from_fn(|index| index as u8);
        let mut reader = Lucene90StoredFieldsFormat
            .fields_reader(StoredFieldsReadState {
                fdt,
                fdx,
                fdm,
                id: &id,
                suffix: "",
            })
            .unwrap();
        reader.close().unwrap();
        assert_eq!(
            reader.check_integrity().unwrap_err().to_string(),
            "closed stored fields reader"
        );
        struct Visitor;
        impl StoredFieldVisitor for Visitor {
            fn needs_field(&mut self, _: i32) -> stored_fields::Result<stored_fields::VisitStatus> {
                panic!("closed reader called visitor")
            }
        }
        assert_eq!(
            reader
                .visit_document(0, &mut Visitor)
                .unwrap_err()
                .to_string(),
            "closed stored fields reader"
        );
        assert!(catch_unwind(AssertUnwindSafe(|| reader.clone_reader())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| reader.get_merge_instance())).is_err());
    }
}
