//! Java's `CrankyNormsFormat` and nested `CrankyNormsConsumer`: guard consumer
//! creation and field addition before delegation; guard close after delegation.
//! Producer creation delegates unchanged. Every guard draws `nextInt(100)` once.
//!
//! The production Lucene90 adapter uses `norms::write_fields` and the real norms
//! reader over directory bytes. Rust's whole-pair encoder requires owned field
//! buffers until close, rather than Java's streaming producer callbacks. Outputs
//! open in Java's order (data, metadata), but encoding and write errors happen at
//! close. This is an explicit format seam, without Codec/IndexWriter routing or
//! IOContext. Single-threaded `Rc<RefCell<JavaRandom>>` replaces Java's Random.
//! Closed consumers allow repeated close; add after close returns a typed I/O
//! error in place of Java's NullPointerException. Valid fields must have unique
//! nonnegative numbers and valid document IDs, as the production encoder requires.
//! This byte-level state has no FieldInfos: callers needing the Java producer's
//! field-metadata cross-check must call `norms::validate_fields` on the metadata.

use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::norms::{self, Norms, NormsField};
use lucene_store::codec_util;
use lucene_store::{DataOutput, Directory, FsIndexOutput};
use lucene_util::java_random::JavaRandom;

/// Exact Java opening guard error.
pub const FAKE_OPEN_ERROR: &str = "Fake IOException from NormsFormat.normsConsumer()";
/// Exact Java field guard error.
pub const FAKE_ADD_ERROR: &str = "Fake IOException from NormsConsumer.addNormsField()";
/// Exact Java closing guard error.
pub const FAKE_CLOSE_ERROR: &str = "Fake IOException from NormsConsumer.close()";

fn error(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> norms::Error {
    lucene_store::Error::Io(std::io::Error::other(message)).into()
}

/// The segment state used by the production norms byte functions.
pub struct NormsState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub max_doc: i32,
}

/// Minimal consumer callbacks, with borrowed values replacing a Java producer.
pub trait NormsConsumer {
    /// Java: `NormsConsumer.addNormsField`.
    fn add_norms_field(&mut self, field: NormsField<'_>) -> norms::Result<()>;
    /// Java: `NormsConsumer.close`.
    fn close(&mut self) -> norms::Result<()>;
}

/// The producer type is preserved by the cranky format without wrapping it.
pub trait NormsFormat {
    type Consumer: NormsConsumer;
    type Producer;
    /// Java: `NormsFormat.normsConsumer`.
    fn norms_consumer(&mut self, state: &NormsState<'_>) -> norms::Result<Self::Consumer>;
    /// Java: `NormsFormat.normsProducer`.
    fn norms_producer(&mut self, state: &NormsState<'_>) -> norms::Result<Self::Producer>;
}

/// Java's cranky format; consumer and format share the same random stream.
pub struct CrankyNormsFormat<F> {
    delegate: F,
    random: Rc<RefCell<JavaRandom>>,
}

impl<F> CrankyNormsFormat<F> {
    pub fn new(delegate: F, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &F {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut F {
        &mut self.delegate
    }
}

impl<F: NormsFormat> NormsFormat for CrankyNormsFormat<F> {
    type Consumer = CrankyNormsConsumer<F::Consumer>;
    type Producer = F::Producer;

    fn norms_consumer(&mut self, state: &NormsState<'_>) -> norms::Result<Self::Consumer> {
        if self.random.borrow_mut().next_int_bounded(100) == 0 {
            return Err(error(FAKE_OPEN_ERROR));
        }
        Ok(CrankyNormsConsumer::new(
            self.delegate.norms_consumer(state)?,
            self.random.clone(),
        ))
    }

    fn norms_producer(&mut self, state: &NormsState<'_>) -> norms::Result<Self::Producer> {
        self.delegate.norms_producer(state)
    }
}

/// Java's nested consumer. It deliberately has no independent closed flag:
/// every successful delegate close, including a repeated one, draws again.
pub struct CrankyNormsConsumer<C> {
    delegate: C,
    random: Rc<RefCell<JavaRandom>>,
}

impl<C> CrankyNormsConsumer<C> {
    pub fn new(delegate: C, random: Rc<RefCell<JavaRandom>>) -> Self {
        Self { delegate, random }
    }
    pub fn delegate(&self) -> &C {
        &self.delegate
    }
    pub fn delegate_mut(&mut self) -> &mut C {
        &mut self.delegate
    }
}

impl<C: NormsConsumer> NormsConsumer for CrankyNormsConsumer<C> {
    fn add_norms_field(&mut self, field: NormsField<'_>) -> norms::Result<()> {
        if self.random.borrow_mut().next_int_bounded(100) == 0 {
            return Err(error(FAKE_ADD_ERROR));
        }
        self.delegate.add_norms_field(field)
    }

    fn close(&mut self) -> norms::Result<()> {
        self.delegate.close()?;
        if self.random.borrow_mut().next_int_bounded(100) == 0 {
            return Err(error(FAKE_CLOSE_ERROR));
        }
        Ok(())
    }
}

/// Concrete production format, with real `.nvm` and `.nvd` directory I/O.
pub struct Lucene90NormsFormat;

enum OwnedField {
    Dense(i32, Vec<i64>),
    Sparse(i32, Vec<(i32, i64)>),
}

/// Owns the output handles and added fields until explicit close.
pub struct Lucene90NormsConsumer {
    outputs: Option<(FsIndexOutput, FsIndexOutput)>,
    fields: Vec<OwnedField>,
    id: [u8; 16],
    suffix: String,
    max_doc: i32,
}

/// Parsed production metadata with owned data bytes and the segment document bound.
pub struct Lucene90NormsProducer {
    pub metadata: Norms,
    data: Vec<u8>,
    max_doc: i32,
}

impl Lucene90NormsProducer {
    /// Java: `NormsProducer.checkIntegrity`, including the data checksum.
    pub fn check_integrity(&self) -> norms::Result<()> {
        let end = self
            .data
            .len()
            .checked_sub(codec_util::FOOTER_LENGTH)
            .ok_or_else(|| error("missing norms data footer"))?;
        codec_util::check_whole_file_footer(&self.data, end)?;
        Ok(())
    }
    pub fn value(&self, field: i32, doc: i32) -> norms::Result<Option<i64>> {
        if doc < 0 || doc >= self.max_doc {
            return Err(norms::Error::DocOutOfRange(doc, self.max_doc));
        }
        let entry = self
            .metadata
            .entry(field)
            .ok_or(norms::Error::UnknownFieldNumber(field))?;
        norms::norm_value(&self.data, entry, doc)
    }
}

fn file_name(state: &NormsState<'_>, extension: &str) -> String {
    if state.suffix.is_empty() {
        format!("{}.{}", state.segment, extension)
    } else {
        format!("{}_{}.{}", state.segment, state.suffix, extension)
    }
}

impl NormsFormat for Lucene90NormsFormat {
    type Consumer = Lucene90NormsConsumer;
    type Producer = Lucene90NormsProducer;

    fn norms_consumer(&mut self, state: &NormsState<'_>) -> norms::Result<Self::Consumer> {
        let mut data = state.directory.create_output(&file_name(state, "nvd"))?;
        let meta = match state.directory.create_output(&file_name(state, "nvm")) {
            Ok(meta) => meta,
            Err(failure) => {
                // Java's failed constructor closes its already-open data output.
                let mut bytes = Vec::new();
                codec_util::write_index_header(
                    &mut bytes,
                    "Lucene90NormsData",
                    0,
                    state.id,
                    state.suffix,
                );
                codec_util::write_footer(&mut bytes);
                data.write_bytes(&bytes);
                let _ = data.close();
                return Err(failure.into());
            }
        };
        Ok(Lucene90NormsConsumer {
            outputs: Some((data, meta)),
            fields: Vec::new(),
            id: *state.id,
            suffix: state.suffix.into(),
            max_doc: state.max_doc,
        })
    }

    fn norms_producer(&mut self, state: &NormsState<'_>) -> norms::Result<Self::Producer> {
        let meta = state.directory.open(&file_name(state, "nvm"))?;
        let (_, metadata) = norms::parse_meta(&meta, state.id, state.suffix)?;
        let data = state.directory.open(&file_name(state, "nvd"))?;
        norms::check_data_header_footer(&data, state.id, state.suffix)?;
        Ok(Lucene90NormsProducer {
            metadata,
            data: data.to_vec(),
            max_doc: state.max_doc,
        })
    }
}

impl NormsConsumer for Lucene90NormsConsumer {
    fn add_norms_field(&mut self, field: NormsField<'_>) -> norms::Result<()> {
        if self.outputs.is_none() {
            return Err(error("closed consumer"));
        }
        self.fields.push(match field {
            NormsField::Dense(number, values) => OwnedField::Dense(number, values.to_vec()),
            NormsField::Sparse(number, values) => OwnedField::Sparse(number, values.to_vec()),
        });
        Ok(())
    }

    fn close(&mut self) -> norms::Result<()> {
        let Some((mut data, mut meta)) = self.outputs.take() else {
            return Ok(());
        };
        let fields: Vec<_> = self
            .fields
            .iter()
            .map(|field| match field {
                OwnedField::Dense(n, v) => NormsField::Dense(*n, v),
                OwnedField::Sparse(n, v) => NormsField::Sparse(*n, v),
            })
            .collect();
        let encoded = if fields.is_empty() {
            // Java closes an unused consumer by writing headers, EOF and footers.
            let mut meta = Vec::new();
            let mut data = Vec::new();
            codec_util::write_index_header(
                &mut meta,
                "Lucene90NormsMetadata",
                0,
                &self.id,
                &self.suffix,
            );
            codec_util::write_index_header(
                &mut data,
                "Lucene90NormsData",
                0,
                &self.id,
                &self.suffix,
            );
            meta.write_i32(-1);
            codec_util::write_footer(&mut meta);
            codec_util::write_footer(&mut data);
            Ok((meta, data))
        } else {
            norms::write_fields(&fields, self.max_doc, &self.id, &self.suffix).map_err(error)
        };
        let (meta_bytes, data_bytes) = match encoded {
            Ok(bytes) => bytes,
            Err(failure) => {
                // Encoding is deferred to close in this adapter. An encoding
                // failure still closes both outputs; cleanup cannot replace it.
                let _ = data.close();
                let _ = meta.close();
                self.fields.clear();
                return Err(failure);
            }
        };
        data.write_bytes(&data_bytes);
        meta.write_bytes(&meta_bytes);
        // Attempt both closes even when the first fails, as IOUtils.close does.
        let data_result = data.close();
        let meta_result = meta.close();
        self.fields.clear();
        data_result?;
        meta_result?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
