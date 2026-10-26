//! Optional same-wire native postings consumers, one per per-field group.
//!
//! Java: `Lucene104PostingsFormat.fieldsConsumer`,
//! `Lucene103BlockTreeTermsWriter.write/close`, `FieldsConsumer.merge`.
//! Inputs borrow the existing flat term buffers; only reference arrays erase
//! their concrete source types. Source integrity checks and document mapping
//! remain index-owned, before the merge callback. Norms are dense snapshots.
//! The native encoder buffers complete files: publication happens during write,
//! including final metadata/footer bytes which Java emits at close. Sticky I/O
//! errors are returned by close. Group ownership, cleanup and synchronization
//! belong to the caller; all consumers stay open until the outer group closes.

use std::{fmt, io};

use lucene_store::{DataOutput, Directory, FsIndexOutput};

use crate::field_infos::{self, FieldInfo};
use crate::per_field_postings::Lucene104PostingsFormat;
use crate::postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource};

/// Native segment identity, full metadata and the selected group's settings.
#[derive(Clone, Copy)]
pub struct PostingsWriteState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub max_doc: i32,
    pub field_infos: &'a [FieldInfo],
    pub format: Lucene104PostingsFormat,
}

/// Operation-local `FieldsConsumer` over native borrowed postings.
pub trait PostingsWriteSink {
    fn write(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()>;

    /// Java's inherited merge ends with a virtual write of mapped fields.
    /// Source acquisition/integrity/map callbacks have already run here.
    fn merge(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.write(inputs, norms)
    }

    fn close(&mut self) -> postings_writer::Result<()>;
}

/// Shared codec configuration opening one native group consumer.
pub trait PostingsWriterFactory: Send + Sync + fmt::Debug {
    fn fields_consumer<'a>(
        &self,
        state: PostingsWriteState<'a>,
    ) -> postings_writer::Result<Box<dyn PostingsWriteSink + 'a>>;
}

fn lifecycle_error(message: &str) -> postings_writer::Error {
    lucene_store::Error::Io(io::Error::new(io::ErrorKind::InvalidInput, message)).into()
}

/// Files in Java constructor order; optional proximity files use all fields.
pub fn output_files(state: PostingsWriteState<'_>) -> Vec<String> {
    let stem = if state.suffix.is_empty() {
        state.segment.to_owned()
    } else {
        format!("{}_{}", state.segment, state.suffix)
    };
    let mut exts = vec!["psm", "doc"];
    if field_infos::has_prox(state.field_infos) {
        exts.push("pos");
        if field_infos::has_payloads_or_offsets(state.field_infos) {
            exts.push("pay");
        }
    }
    exts.extend(["tim", "tip", "tmd"]);
    exts.into_iter()
        .map(|ext| format!("{stem}.{ext}"))
        .collect()
}

/// Native encoding and output lifecycle without replacing the reader graph.
pub struct NativePostingsWriter<'a> {
    state: PostingsWriteState<'a>,
    outputs: Vec<(String, Option<FsIndexOutput>)>,
    written: bool,
    closed: bool,
}

impl<'a> NativePostingsWriter<'a> {
    /// Java opens postings metadata/data first, then block-tree outputs.
    /// A failed acquisition closes all previously acquired outputs.
    pub fn new(state: PostingsWriteState<'a>) -> postings_writer::Result<Self> {
        let mut writer = Self {
            state,
            outputs: Vec::new(),
            written: false,
            closed: false,
        };
        for name in output_files(state) {
            match state.directory.create_output(&name) {
                Ok(output) => writer.outputs.push((name, Some(output))),
                Err(error) => {
                    let _ = writer.close();
                    return Err(error.into());
                }
            }
        }
        Ok(writer)
    }
}

impl PostingsWriteSink for NativePostingsWriter<'_> {
    fn write(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        if self.closed || self.written {
            return Err(lifecycle_error(
                "postings consumer is closed or already written",
            ));
        }
        self.written = true;
        let options = postings_writer::WriteOptions {
            has_prox: Some(field_infos::has_prox(self.state.field_infos)),
            has_payloads_or_offsets: Some(field_infos::has_payloads_or_offsets(
                self.state.field_infos,
            )),
            ..self.state.format.write_options()
        };
        let populated: Vec<_> = inputs
            .iter()
            .copied()
            .filter(|i| !i.terms.is_empty())
            .collect();
        let bytes = if populated.is_empty() {
            postings_writer::write_no_fields(&options, self.state.id, self.state.suffix)?
        } else {
            postings_writer::write_fields_with_options(
                &populated,
                norms,
                &options,
                self.state.id,
                self.state.suffix,
            )?
        };
        for (name, output) in &mut self.outputs {
            let data = match name.rsplit('.').next().expect("output extension") {
                "psm" => &bytes.psm,
                "doc" => &bytes.doc,
                "pos" => &bytes.pos,
                "pay" => &bytes.pay,
                "tim" => &bytes.tim,
                "tip" => &bytes.tip,
                "tmd" => &bytes.tmd,
                _ => unreachable!("native output extension"),
            };
            output.as_mut().expect("open output").write_bytes(data);
        }
        Ok(())
    }

    /// Java closes block-tree metadata/terms/index, then postings outputs.
    /// Every output is attempted; the first error remains primary.
    fn close(&mut self) -> postings_writer::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let mut first = None;
        for ext in ["tmd", "tim", "tip", "psm", "doc", "pos", "pay"] {
            for (name, output) in &mut self.outputs {
                if name.rsplit('.').next() == Some(ext) {
                    if let Some(output) = output.take() {
                        if let Err(error) = output.close() {
                            first.get_or_insert(error);
                        }
                    }
                }
            }
        }
        first.map_or(Ok(()), |error| Err(error.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::IndexOptions;
    use crate::postings_writer::{TermPostings, WriteOptions};
    use lucene_store::ByteBuffersDirectory;

    const ID: [u8; 16] = [63; 16];

    fn state<'a>(directory: &'a dyn Directory, fields: &'a [FieldInfo]) -> PostingsWriteState<'a> {
        PostingsWriteState {
            directory,
            segment: "_p",
            id: &ID,
            suffix: "Lucene104_0",
            max_doc: 2,
            field_infos: fields,
            format: Lucene104PostingsFormat::default(),
        }
    }

    #[test]
    fn native_bytes_and_block_settings_survive_borrowed_sources() {
        let mut field = FieldInfo::new("body", 0);
        field.index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        field.store_payloads = true;
        let fields = [field];
        let terms: Vec<_> = (0..12)
            .map(|i| TermPostings {
                term: format!("t{i:02}").into_bytes(),
                docs: vec![(0, 2), (1, 1)],
                positions: vec![vec![0, 0], vec![2]],
                offsets: vec![vec![(0, 1), (0, 2)], vec![(3, 4)]],
                payload_bytes: vec![7, 8, 9],
                payload_lengths: vec![1, 0, 2],
            })
            .collect();
        let references: Vec<&dyn TermSource> = terms.iter().map(|t| t as &dyn TermSource).collect();
        let inputs = [FieldPostingsInput {
            field_number: 0,
            index_options: fields[0].index_options,
            doc_count: 2,
            has_payloads: true,
            terms: references.as_slice(),
        }];
        let values = [2, 4];
        let norms = [FieldNorms {
            field_number: 0,
            values: &values,
        }];
        for suffix in ["", "Lucene104_0"] {
            for format in [
                Lucene104PostingsFormat::default(),
                Lucene104PostingsFormat::new(2, 3).unwrap(),
            ] {
                let directory = ByteBuffersDirectory::new();
                let state = PostingsWriteState {
                    suffix,
                    format,
                    ..state(&directory, &fields)
                };
                let mut consumer = NativePostingsWriter::new(state).unwrap();
                consumer.write(&inputs, &norms).unwrap();
                assert_eq!(
                    consumer.write(&inputs, &norms).unwrap_err().to_string(),
                    "postings consumer is closed or already written"
                );
                consumer.close().unwrap();
                consumer.close().unwrap();
                let options = WriteOptions {
                    has_prox: Some(true),
                    has_payloads_or_offsets: Some(true),
                    ..format.write_options()
                };
                let expected = postings_writer::write_fields_with_options(
                    &inputs, &norms, &options, &ID, suffix,
                )
                .unwrap();
                let stem = if suffix.is_empty() {
                    "_p"
                } else {
                    "_p_Lucene104_0"
                };
                for (ext, bytes) in [
                    ("doc", expected.doc),
                    ("psm", expected.psm),
                    ("pos", expected.pos),
                    ("pay", expected.pay),
                    ("tim", expected.tim),
                    ("tip", expected.tip),
                    ("tmd", expected.tmd),
                ] {
                    assert_eq!(&*directory.open(&format!("{stem}.{ext}")).unwrap(), bytes);
                }
            }
        }
    }

    #[test]
    fn empty_groups_keep_segment_wide_proximity_files_and_merge_finishes() {
        for options in [
            IndexOptions::Docs,
            IndexOptions::DocsAndFreqsAndPositions,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
        ] {
            let directory = ByteBuffersDirectory::new();
            let mut field = FieldInfo::new("empty", 0);
            field.index_options = options;
            let fields = [field];
            let state = state(&directory, &fields);
            let mut consumer = NativePostingsWriter::new(state).unwrap();
            consumer.merge(&[], &[]).unwrap();
            consumer.close().unwrap();
            let names = directory.list_all().unwrap();
            assert_eq!(
                names.len(),
                if options.subsumes_offsets() {
                    7
                } else if options.subsumes_positions() {
                    6
                } else {
                    5
                }
            );
            assert_eq!(
                names.contains(&"_p_Lucene104_0.pos".into()),
                options.subsumes_positions()
            );
            assert_eq!(
                names.contains(&"_p_Lucene104_0.pay".into()),
                options.subsumes_offsets()
            );
            for name in names {
                assert!(directory.file_length(&name).unwrap() > 0);
            }
        }
    }

    #[test]
    fn construction_preserves_collisions_and_failed_writes_are_terminal() {
        let mut field = FieldInfo::new("body", 0);
        field.index_options = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;
        let fields = [field];
        let directory = ByteBuffersDirectory::new();
        for occupied in output_files(state(&directory, &fields)) {
            let directory = ByteBuffersDirectory::new();
            let mut output = directory.create_output(&occupied).unwrap();
            output.write_byte(19);
            output.close().unwrap();
            let error = NativePostingsWriter::new(state(&directory, &fields))
                .err()
                .unwrap();
            assert!(
                matches!(error, postings_writer::Error::Store(lucene_store::Error::Io(ref e)) if e.kind() == io::ErrorKind::AlreadyExists)
            );
            assert_eq!(&*directory.open(&occupied).unwrap(), [19]);
        }
        let directory = ByteBuffersDirectory::new();
        let mut consumer = NativePostingsWriter::new(state(&directory, &fields)).unwrap();
        consumer.close().unwrap();
        assert!(consumer.write(&[], &[]).is_err());
        let directory = ByteBuffersDirectory::new();
        let mut consumer = NativePostingsWriter::new(state(&directory, &fields)).unwrap();
        let invalid = TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 0)],
            ..Default::default()
        };
        let terms = [&invalid as &dyn TermSource];
        let inputs = [FieldPostingsInput {
            field_number: 0,
            index_options: fields[0].index_options,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        }];
        assert!(consumer.write(&inputs, &[]).is_err());
        assert!(consumer.write(&[], &[]).is_err());
        consumer.close().unwrap();
    }
}
