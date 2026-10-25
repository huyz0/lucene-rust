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
//! IndexWriter codec selection is not implemented here.

use std::collections::BTreeMap;

use lucene_codecs::blockterms::{
    self, fixed_gap::DEFAULT_TERM_INDEX_INTERVAL, BlockTermsReader, Error,
    EveryNOrDocFreqTermSelector, EveryNTermSelector, FixedGapTermsIndexReader,
    FixedGapTermsIndexWriter, Result, TermsIndexReader, VariableGapTermsIndexReader,
    VariableGapTermsIndexWriter,
};
use lucene_codecs::postings_writer::{FieldNorms, FieldPostingsInput, TermSource};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};

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
        let (postings, tib, indexes) = match self {
            Self::FixedGap {
                term_index_interval,
            } => {
                let index = FixedGapTermsIndexWriter::new(state, term_index_interval)?;
                let out = blockterms::write(index, inputs, norms, state)?;
                (out.postings, out.tib, vec![("tii", out.index)])
            }
            Self::VarGapFixedInterval {
                term_index_interval,
            } => {
                let index = VariableGapTermsIndexWriter::new(
                    state,
                    EveryNTermSelector::new(term_index_interval),
                );
                let out = blockterms::write(index, inputs, norms, state)?;
                (
                    out.postings,
                    out.tib,
                    vec![("tmv", out.index.tmv), ("tiv", out.index.tiv)],
                )
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
                (
                    out.postings,
                    out.tib,
                    vec![("tmv", out.index.tmv), ("tiv", out.index.tiv)],
                )
            }
        };
        let mut files: BTreeMap<String, Vec<u8>> = indexes
            .into_iter()
            .map(|(ext, bytes)| (ext.to_string(), bytes))
            .collect();
        files.insert("tib".to_string(), tib);
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
