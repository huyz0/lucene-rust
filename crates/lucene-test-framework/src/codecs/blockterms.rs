//! `LuceneFixedGap`, `LuceneVarGapFixedInterval` and
//! `LuceneVarGapDocFreqInterval`. All three use Lucene104 postings and the
//! block-terms dictionary; only the terms index differs. Readers need no
//! constructor settings: the index files contain the selected terms.
//!
//! Java: each format's `fieldsProducer` and `fieldsConsumer`. Rust passes
//! whole file buffers rather than a Directory, returns the existing codec
//! reader directly, and writes supplied postings in one call. The caller
//! supplies fields in name order, as Java's Fields iterator does. Resource
//! cleanup uses ownership instead of Java's close-on-failure blocks.
//! IndexWriter routes a field to one through
//! `Lucene104PostingsFormat::dictionary` with the statics below, once
//! [`super::register_postings_formats`] has made them readable by name.

use std::collections::BTreeMap;

use lucene_codecs::blockterms::{
    self, fixed_gap::DEFAULT_TERM_INDEX_INTERVAL, BlockTermsReader, Error,
    EveryNOrDocFreqTermSelector, EveryNTermSelector, FixedGapTermsIndexReader,
    FixedGapTermsIndexWriter, Result, TermsIndexReader, VariableGapTermsIndexReader,
    VariableGapTermsIndexWriter,
};
use lucene_codecs::blocktree;
use lucene_codecs::postings_writer::{
    self, DictionaryField, DictionaryTerm, FieldNorms, FieldPostingsInput, PostingsFiles,
    TermSource, WriteOptions,
};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_codecs::terms_dictionary::{ErasedInput, TermsDictionaryFormat};

/// The test-framework's three block-terms postings formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTermsFormat {
    /// `LuceneFixedGap`: fixed gap, with ordinal seeks.
    FixedGap {
        /// `LuceneFixedGap(int termIndexInterval)`.
        term_index_interval: i32,
    },
    /// `LuceneVarGapFixedInterval`: every Nth term in an FST.
    VarGapFixedInterval {
        /// `LuceneVarGapFixedInterval(int termIndexInterval)`.
        term_index_interval: i32,
    },
    /// `LuceneVarGapDocFreqInterval`: every Nth or high-frequency term.
    VarGapDocFreqInterval {
        /// `LuceneVarGapDocFreqInterval`'s first constructor argument.
        doc_freq_threshold: i32,
        /// `LuceneVarGapDocFreqInterval`'s second constructor argument.
        term_index_interval: i32,
    },
}

impl BlockTermsFormat {
    /// `new LuceneFixedGap()`: an index term every 32 terms.
    pub fn fixed_gap() -> Self {
        Self::FixedGap {
            term_index_interval: DEFAULT_TERM_INDEX_INTERVAL,
        }
    }

    /// `new LuceneVarGapFixedInterval()`: an index term every 32 terms.
    pub fn var_gap_fixed_interval() -> Self {
        Self::VarGapFixedInterval {
            term_index_interval: DEFAULT_TERM_INDEX_INTERVAL,
        }
    }

    /// `new LuceneVarGapDocFreqInterval()`: every 32 terms or docFreq at
    /// least 1,000,000.
    pub fn var_gap_doc_freq_interval() -> Self {
        Self::VarGapDocFreqInterval {
            doc_freq_threshold: 1_000_000,
            term_index_interval: DEFAULT_TERM_INDEX_INTERVAL,
        }
    }

    /// Java's SPI name, also used in the per-field segment suffix.
    pub fn name(self) -> &'static str {
        match self {
            Self::FixedGap { .. } => "LuceneFixedGap",
            Self::VarGapFixedInterval { .. } => "LuceneVarGapFixedInterval",
            Self::VarGapDocFreqInterval { .. } => "LuceneVarGapDocFreqInterval",
        }
    }

    /// Java: `fieldsProducer`. Files are keyed by extension (`tib`, `tii`,
    /// `tmv`, `tiv`); postings buffers are passed to the returned enum's
    /// `postings` method, as for `BlockTermsReader`.
    pub fn fields_producer(
        self,
        files: &BTreeMap<String, Vec<u8>>,
        state: &SegmentReadState<'_>,
    ) -> Result<BlockTermsReader> {
        let file = |extension: &str| {
            files.get(extension).ok_or_else(|| {
                Error::IllegalArgument(format!("missing {extension} file for {}", self.name()))
            })
        };
        let index: TermsIndexReader = match self {
            Self::FixedGap { .. } => FixedGapTermsIndexReader::open(file("tii")?, state)?.into(),
            Self::VarGapFixedInterval { .. } | Self::VarGapDocFreqInterval { .. } => {
                VariableGapTermsIndexReader::open(file("tmv")?, file("tiv")?, state)?.into()
            }
        };
        BlockTermsReader::open(file("tib")?.clone(), index, state)
    }

    /// Java: `fieldsConsumer(state).write(fields, norms)` then `close()`.
    /// Returns dictionary, terms-index and Lucene104 postings files keyed
    /// by extension, using this format's constructor settings.
    ///
    /// # Errors
    /// The underlying terms-index and postings writers' input errors.
    pub fn fields_consumer<T: TermSource>(
        self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        let (postings, dictionary) = self.write_parts(inputs, norms, state)?;
        let mut files: BTreeMap<String, Vec<u8>> = self
            .dictionary_extensions()
            .iter()
            .zip(dictionary)
            .map(|(ext, bytes)| (ext.to_string(), bytes))
            .collect();
        files.insert("doc".to_string(), postings.doc);
        files.insert("psm".to_string(), postings.psm);
        if !postings.pos.is_empty() {
            files.insert("pos".to_string(), postings.pos);
        }
        if !postings.pay.is_empty() {
            files.insert("pay".to_string(), postings.pay);
        }
        Ok(files)
    }

    /// The dictionary's files: `.tib`, then the terms index's.
    fn dictionary_extensions(self) -> &'static [&'static str] {
        match self {
            Self::FixedGap { .. } => &["tib", "tii"],
            Self::VarGapFixedInterval { .. } | Self::VarGapDocFreqInterval { .. } => {
                &["tib", "tmv", "tiv"]
            }
        }
    }

    /// The postings, and the dictionary's files in
    /// [`Self::dictionary_extensions`] order.
    fn write_parts<T: TermSource>(
        self,
        inputs: &[FieldPostingsInput<'_, T>],
        norms: &[FieldNorms<'_>],
        state: &SegmentWriteState<'_>,
    ) -> Result<(PostingsFiles, Vec<Vec<u8>>)> {
        Ok(match self {
            Self::FixedGap {
                term_index_interval,
            } => {
                let index = FixedGapTermsIndexWriter::new(state, term_index_interval)?;
                let out = blockterms::write(index, inputs, norms, state)?;
                (out.postings, vec![out.tib, out.index])
            }
            Self::VarGapFixedInterval {
                term_index_interval,
            } => {
                let index = VariableGapTermsIndexWriter::new(
                    state,
                    EveryNTermSelector::new(term_index_interval),
                );
                let out = blockterms::write(index, inputs, norms, state)?;
                (out.postings, vec![out.tib, out.index.tmv, out.index.tiv])
            }
            Self::VarGapDocFreqInterval {
                doc_freq_threshold,
                term_index_interval,
            } => {
                let index = VariableGapTermsIndexWriter::new(
                    state,
                    EveryNOrDocFreqTermSelector::new(doc_freq_threshold, term_index_interval),
                );
                let out = blockterms::write(index, inputs, norms, state)?;
                (out.postings, vec![out.tib, out.index.tmv, out.index.tiv])
            }
        })
    }
}

/// `new LuceneFixedGap()`, as IndexWriter routes fields to it
/// ([`lucene_codecs::per_field_postings::Lucene104PostingsFormat::dictionary`])
/// and reads its groups back.
pub static FIXED_GAP: BlockTermsFormat = BlockTermsFormat::FixedGap {
    term_index_interval: DEFAULT_TERM_INDEX_INTERVAL,
};

/// `new LuceneVarGapFixedInterval()`.
pub static VAR_GAP_FIXED_INTERVAL: BlockTermsFormat = BlockTermsFormat::VarGapFixedInterval {
    term_index_interval: DEFAULT_TERM_INDEX_INTERVAL,
};

/// `new LuceneVarGapDocFreqInterval()`.
pub static VAR_GAP_DOC_FREQ_INTERVAL: BlockTermsFormat = BlockTermsFormat::VarGapDocFreqInterval {
    doc_freq_threshold: 1_000_000,
    term_index_interval: DEFAULT_TERM_INDEX_INTERVAL,
};

impl TermsDictionaryFormat for BlockTermsFormat {
    fn name(&self) -> &'static str {
        BlockTermsFormat::name(*self)
    }

    fn extensions(&self) -> &'static [&'static str] {
        self.dictionary_extensions()
    }

    fn write(
        &self,
        inputs: &[ErasedInput<'_, '_>],
        norms: &[FieldNorms<'_>],
        _options: &WriteOptions,
        state: &SegmentWriteState<'_>,
    ) -> postings_writer::Result<(PostingsFiles, Vec<Vec<u8>>)> {
        self.write_parts(inputs, norms, state).map_err(|e| match e {
            Error::Write(e) => e,
            Error::Store(e) => postings_writer::Error::Store(e),
            e => postings_writer::Error::Dictionary(e.to_string()),
        })
    }

    fn read(
        &self,
        files: &[&[u8]],
        state: &SegmentReadState<'_>,
    ) -> blocktree::Result<Vec<DictionaryField>> {
        let name = BlockTermsFormat::name(*self);
        let err = |e: Error| match e {
            Error::Store(e) => blocktree::Error::Store(e),
            Error::Postings(e) => blocktree::Error::Postings(e),
            e => blocktree::Error::ForeignDictionary {
                format: name,
                message: e.to_string(),
            },
        };
        let files: BTreeMap<String, Vec<u8>> = self
            .dictionary_extensions()
            .iter()
            .zip(files)
            .map(|(ext, bytes)| (ext.to_string(), bytes.to_vec()))
            .collect();
        let reader = self.fields_producer(&files, state).map_err(err)?;
        let mut fields = Vec::with_capacity(reader.size());
        for field_name in reader.field_names() {
            let Some(terms) = reader.terms(field_name) else {
                continue;
            };
            let info = terms.info();
            let mut field = DictionaryField {
                field_number: info.number,
                index_options: info.index_options,
                has_payloads: info.store_payloads,
                doc_count: terms.doc_count(),
                terms: Vec::new(),
            };
            let mut it = terms.iterator();
            while let Some(bytes) = it.next().map_err(err)? {
                let bytes = bytes.to_vec();
                let state = it.term_state().map_err(err)?;
                field.terms.push(DictionaryTerm {
                    bytes,
                    doc_freq: state.doc_freq,
                    total_term_freq: state.total_term_freq,
                    meta: state.meta,
                });
            }
            fields.push(field);
        }
        Ok(fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
    use lucene_codecs::postings_writer::TermPostings;

    #[test]
    fn writer_errors_propagate_and_optional_postings_files_follow_field_options() {
        let infos = FieldInfos::new(vec![
            FieldInfo::new("body", 0).with_index_options(IndexOptions::DocsAndFreqs)
        ])
        .unwrap();
        let state = SegmentWriteState {
            max_doc: 1,
            segment_id: &[7; 16],
            segment_suffix: "",
            field_infos: &infos,
        };
        let mut terms = [TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 0)],
            ..TermPostings::default()
        }];
        for format in [
            BlockTermsFormat::fixed_gap(),
            BlockTermsFormat::var_gap_fixed_interval(),
            BlockTermsFormat::var_gap_doc_freq_interval(),
        ] {
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
                has_payloads: false,
                terms: &terms,
            };
            assert!(matches!(
                format.fields_consumer(&[input], &[], &state),
                Err(Error::Write(_))
            ));
        }
        assert!(matches!(
            (BlockTermsFormat::FixedGap {
                term_index_interval: 0
            })
            .fields_consumer::<TermPostings>(&[], &[], &state),
            Err(Error::IllegalArgument(_))
        ));
        terms[0].docs[0].1 = 1;
        let files = BlockTermsFormat::fixed_gap()
            .fields_consumer(
                &[FieldPostingsInput {
                    field_number: 0,
                    index_options: IndexOptions::DocsAndFreqs,
                    doc_count: 1,
                    has_payloads: false,
                    terms: &terms,
                }],
                &[],
                &state,
            )
            .unwrap();
        assert_eq!(
            files.keys().map(String::as_str).collect::<Vec<_>>(),
            ["doc", "psm", "tib", "tii"]
        );

        let infos = FieldInfos::new(vec![FieldInfo::new("body", 0)
            .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)])
        .unwrap();
        let state = SegmentWriteState {
            field_infos: &infos,
            ..state
        };
        terms[0].positions = vec![vec![0]];
        terms[0].offsets = vec![vec![(0, 1)]];
        let files = BlockTermsFormat::fixed_gap()
            .fields_consumer(
                &[FieldPostingsInput {
                    field_number: 0,
                    index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
                    doc_count: 1,
                    has_payloads: false,
                    terms: &terms,
                }],
                &[],
                &state,
            )
            .unwrap();
        assert_eq!(
            files.keys().map(String::as_str).collect::<Vec<_>>(),
            ["doc", "pay", "pos", "psm", "tib", "tii"]
        );
        for extension in ["doc", "pos", "pay", "psm", "tib", "tii"] {
            let bytes = &files[extension];
            lucene_store::codec_util::check_whole_file_footer(
                bytes,
                bytes.len() - lucene_store::codec_util::FOOTER_LENGTH,
            )
            .unwrap();
        }
    }

    #[test]
    fn missing_index_reports_the_required_extension_and_format() {
        let infos = FieldInfos::new(vec![]).unwrap();
        let state = SegmentReadState {
            max_doc: 0,
            segment_id: &[7; 16],
            segment_suffix: "",
            field_infos: &infos,
        };
        for (format, extension) in [
            (BlockTermsFormat::fixed_gap(), "tii"),
            (BlockTermsFormat::var_gap_fixed_interval(), "tmv"),
            (BlockTermsFormat::var_gap_doc_freq_interval(), "tmv"),
        ] {
            let error = format
                .fields_producer(&BTreeMap::new(), &state)
                .unwrap_err();
            assert!(matches!(error, Error::IllegalArgument(_)));
            assert_eq!(
                error.to_string(),
                format!("missing {extension} file for {}", format.name())
            );
        }
    }
}
