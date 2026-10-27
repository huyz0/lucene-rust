//! The test framework's `CompressingCodec` installed in
//! [`lucene_index::index_writer::IndexWriter`] (`IndexWriter::set_codec`).
//!
//! Java's codec is a `FilterCodec` over the default codec whose stored fields
//! and term vectors are a `Lucene90CompressingStoredFieldsFormat` and a
//! `Lucene90CompressingTermVectorsFormat` built from the same settings. The
//! installed codec keeps every other format native and hands IndexWriter
//! those settings ([`Lucene104Codec::compressing_format`]): flushes and
//! merges write `.fdt`/`.tvd` under the codec's data codec name and segment
//! suffix, with its compression mode, chunk size, documents per chunk and
//! block shift, and merges copy compressed chunks between segments of the
//! same format as Java's do. Segments record the codec's name, which Lucene
//! resolves through its SPI to the same subclass (with its default settings:
//! the reader takes the chunk size and block shift from the files, but a
//! segment suffix only from the codec instance).

use std::fmt;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::compression::{self, CompressingFormat, Compression};

use super::compressing_codec::{CompressingCodec, CompressionMode, InvalidConfiguration};

/// [`CompressingCodec::installed`]: the codec's name, its formats' shared
/// settings, and its Java `toString()`.
#[derive(Debug)]
pub struct InstalledCompressingCodec {
    format: &'static CompressingFormat,
    display: String,
}

impl CompressingCodec {
    /// This codec, installable in IndexWriter.
    ///
    /// The settings outlive every writer and reader they reach (each
    /// installation leaks one small descriptor, as a test codec may).
    ///
    /// # Errors
    /// A `null` compression mode, which Java's formats only reject when a
    /// writer or reader is created, and more than
    /// [`compression::COMPRESSING_MAX_DOCS_PER_CHUNK`] documents per chunk,
    /// which this port's readers bound chunks by.
    pub fn installed(&self) -> Result<InstalledCompressingCodec, InvalidConfiguration> {
        let settings = &self.stored_fields_format().0;
        let Some(mode) = settings.compression_mode.as_ref() else {
            return Err(InvalidConfiguration(
                "a null compression mode cannot write".into(),
            ));
        };
        if settings.max_docs_per_chunk > compression::COMPRESSING_MAX_DOCS_PER_CHUNK {
            return Err(InvalidConfiguration(format!(
                "maxDocsPerChunk must be at most {}, got {}",
                compression::COMPRESSING_MAX_DOCS_PER_CHUNK,
                settings.max_docs_per_chunk
            )));
        }
        let compression = match mode.mode() {
            CompressionMode::Fast => Compression::Fast,
            CompressionMode::FastDecompression => Compression::FastDecompression,
            CompressionMode::HighCompression => Compression::HighCompression,
            CompressionMode::Dummy => Compression::Dummy,
            CompressionMode::DeflateWithPreset => Compression::DeflateWithPreset,
            CompressionMode::Lz4WithPreset => Compression::Lz4WithPreset,
        };
        let leak = |s: &str| -> &'static str { Box::leak(s.into()) };
        let format = Box::leak(Box::new(CompressingFormat {
            format_name: leak(settings.format_name()),
            segment_suffix: leak(settings.segment_suffix()),
            compression,
            chunk_size: settings.chunk_size,
            max_docs_per_chunk: settings.max_docs_per_chunk,
            block_shift: settings.block_shift,
        }));
        // `Codec.forName`: the six subclasses are in the jar's SPI; a
        // custom name is only this port's to read.
        super::register_codecs();
        lucene_codecs::codecs::register(lucene_codecs::codecs::renamed(
            lucene_codecs::codecs::lucene104(),
            format.format_name,
        ));
        Ok(InstalledCompressingCodec {
            format,
            display: self.to_string(),
        })
    }
}

impl InstalledCompressingCodec {
    /// The stored-fields and term-vectors settings.
    pub fn format(&self) -> &'static CompressingFormat {
        self.format
    }
}

impl Lucene104Codec for InstalledCompressingCodec {
    /// `FilterCodec(name, ...)`: the data codec name, which is also the
    /// codec's SPI name (`FastCompressingStoredFieldsData`).
    fn name(&self) -> &str {
        self.format.format_name
    }

    fn compressing_format(&self) -> Option<&'static CompressingFormat> {
        Some(self.format)
    }
}

impl fmt::Display for InstalledCompressingCodec {
    /// `CompressingCodec.toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display)
    }
}

#[cfg(test)]
mod tests {
    use super::super::compressing_codec::{CodecKind, ModeHandle};
    use super::*;

    #[test]
    fn every_kind_installs_with_its_name_suffix_and_settings() {
        for kind in CodecKind::ALL {
            let codec = kind.configured(100, 7, true, 5).unwrap();
            let installed = codec.installed().unwrap();
            let format = installed.format();
            assert_eq!(installed.name(), codec.name());
            assert_eq!(format.format_name, codec.name());
            assert_eq!(
                format.segment_suffix,
                codec.stored_fields_format().0.segment_suffix()
            );
            assert!(!format.segment_suffix.is_empty());
            assert_eq!(
                format.compression.to_string(),
                codec.stored_fields_format().0.mode_display()
            );
            assert_eq!(
                (
                    format.chunk_size,
                    format.max_docs_per_chunk,
                    format.block_shift
                ),
                (100, 7, 5)
            );
            assert_eq!(installed.to_string(), codec.to_string());
        }
    }

    #[test]
    fn a_null_mode_or_an_unreadable_document_cap_is_refused() {
        let null = CompressingCodec::new("NullMode", None, 10, 10, 10).unwrap();
        assert!(null.installed().is_err());
        let huge = CompressingCodec::new(
            "Huge",
            Some(ModeHandle::new(CompressionMode::Fast)),
            10,
            compression::COMPRESSING_MAX_DOCS_PER_CHUNK + 1,
            10,
        )
        .unwrap();
        assert!(huge.installed().is_err());
    }
}
