//! Port of `VersionBlockTreeTermsReader` and `VersionFieldReader`: opens a
//! group's `.tiv`/`.tipv` and loads each field's FST terms index.

use std::collections::BTreeMap;

use lucene_store::codec_util::{self, FOOTER_LENGTH, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::error::Error;

use super::{
    IdVersionSegmentTermsEnum, IndexOutput, IndexOutputs, OUTPUT_FLAGS_NUM_BITS,
    POSTINGS_TERMS_CODEC, TERMS_CODEC_NAME, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT,
};
use crate::fst_util::TypedFst;

/// `VersionBlockTreeTermsReader`: the fields of one group, by name.
#[derive(Debug)]
pub struct VersionBlockTreeTermsReader {
    /// `in`: the whole `.tiv`, which every terms enum reads blocks from.
    pub(super) tiv: Vec<u8>,
    fields: BTreeMap<String, VersionFieldReader>,
}

/// `VersionFieldReader`: one field's statistics and terms index.
#[derive(Debug)]
pub struct VersionFieldReader {
    field_number: i32,
    num_terms: i64,
    root_code: IndexOutput,
    pub(super) root_block_fp: i64,
    index_start_fp: i64,
    min_term: Vec<u8>,
    max_term: Vec<u8>,
    pub(super) index: TypedFst<IndexOutputs>,
}

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Corrupted(msg.into())
}

/// `readBytesRef(in)`.
fn read_bytes_ref(input: &mut SliceInput<'_>) -> Result<Vec<u8>, Error> {
    let n = input.read_vint()?;
    let n = usize::try_from(n).map_err(|_| corrupt(format!("negative length {n}")))?;
    if n > input.len().saturating_sub(input.position()) {
        return Err(corrupt(format!("length {n} past the end")));
    }
    let mut bytes = vec![0u8; n];
    input.read_bytes(&mut bytes)?;
    Ok(bytes)
}

/// `seekDir(input)`: the directory offset in the long before the footer.
pub(super) fn seek_dir(input: &mut SliceInput<'_>, len: usize) -> Result<(), Error> {
    let at = len
        .checked_sub(FOOTER_LENGTH + 8)
        .ok_or_else(|| corrupt("file too short for its directory offset"))?;
    input.seek(at)?;
    let dir = input.read_i64()?;
    let dir = usize::try_from(dir)
        .ok()
        .filter(|&d| d <= at)
        .ok_or_else(|| corrupt(format!("invalid directory offset {dir}")))?;
    input.seek(dir)
}

impl VersionBlockTreeTermsReader {
    /// `new VersionBlockTreeTermsReader(postingsReader, state)`: `field_name`
    /// is `state.fieldInfos.fieldInfo(number).name`, `max_doc` the segment's.
    ///
    /// # Errors
    /// `Corrupted` for a bad header, footer, checksum or field directory,
    /// with Java's messages where Java has one.
    pub fn open(
        tiv: Vec<u8>,
        tipv: &[u8],
        segment_id: &[u8; ID_LENGTH],
        segment_suffix: &str,
        max_doc: i32,
        field_name: &dyn Fn(i32) -> Option<String>,
    ) -> Result<Self, Error> {
        let mut input = SliceInput::new(&tiv);
        let terms_version = codec_util::check_index_header(
            &mut input,
            TERMS_CODEC_NAME,
            VERSION_CURRENT,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?
        .version;
        let mut index_in = SliceInput::new(tipv);
        let index_version = codec_util::check_index_header(
            &mut index_in,
            TERMS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?
        .version;
        if index_version != terms_version {
            return Err(corrupt(format!(
                "mixmatched version files: {terms_version},{index_version}"
            )));
        }
        // `CodecUtil.checksumEntireFile(indexIn)`.
        let index_payload_end = tipv
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(tipv, index_payload_end)?;
        // `postingsReader.init(in, state)`.
        codec_util::check_index_header(
            &mut input,
            POSTINGS_TERMS_CODEC,
            VERSION_CURRENT,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        codec_util::retrieve_checksum(&tiv)?;
        seek_dir(&mut input, tiv.len())?;
        seek_dir(&mut index_in, tipv.len())?;

        let num_fields = input.read_vint()?;
        if num_fields < 0 {
            return Err(corrupt(format!("invalid numFields: {num_fields}")));
        }
        let mut fields = BTreeMap::new();
        for _ in 0..num_fields {
            let field = input.read_vint()?;
            let num_terms = input.read_vlong()?;
            let code = read_bytes_ref(&mut input)?;
            let version = input.read_vlong()?;
            let root_code = IndexOutput {
                first: code,
                second: version,
            };
            let name = field_name(field)
                .ok_or_else(|| corrupt(format!("invalid field number: {field}")))?;
            let min_term = read_bytes_ref(&mut input)?;
            let max_term = read_bytes_ref(&mut input)?;
            // `docCount = (int) numTerms`; sumDocFreq = sumTotalTermFreq =
            // numTerms, so Java's other two checks cannot fail.
            let doc_count = i32::try_from(num_terms).unwrap_or(-1);
            if doc_count < 0 || doc_count > max_doc {
                return Err(corrupt(format!(
                    "invalid docCount: {doc_count} maxDoc: {max_doc}"
                )));
            }
            let index_start_fp = index_in.read_vlong()?;
            let reader = VersionFieldReader::new(
                field,
                num_terms,
                root_code,
                index_start_fp,
                tipv,
                min_term,
                max_term,
            )?;
            if fields.insert(name.clone(), reader).is_some() {
                return Err(corrupt(format!("duplicate field: {name}")));
            }
        }
        Ok(VersionBlockTreeTermsReader { tiv, fields })
    }

    /// `terms(field)`.
    pub fn terms(&self, field: &str) -> Option<&VersionFieldReader> {
        self.fields.get(field)
    }

    /// `iterator()` over the field names, in order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.fields.len()
    }

    /// `checkIntegrity()`: `checksumEntireFile(in)`.
    ///
    /// # Errors
    /// `Corrupted` when the `.tiv` checksum does not match.
    pub fn check_integrity(&self) -> Result<(), Error> {
        let end = self
            .tiv
            .len()
            .checked_sub(FOOTER_LENGTH)
            .ok_or_else(|| corrupt("file too small to contain a codec footer"))?;
        codec_util::check_whole_file_footer(&self.tiv, end).map(|_| ())
    }

    /// `terms(field).iterator()`.
    pub fn iterator(&self, field: &str) -> Option<IdVersionSegmentTermsEnum<'_>> {
        self.fields
            .get(field)
            .map(|fr| IdVersionSegmentTermsEnum::new(&self.tiv, fr))
    }
}

impl VersionFieldReader {
    #[allow(clippy::too_many_arguments)]
    fn new(
        field_number: i32,
        num_terms: i64,
        root_code: IndexOutput,
        index_start_fp: i64,
        tipv: &[u8],
        min_term: Vec<u8>,
        max_term: Vec<u8>,
    ) -> Result<Self, Error> {
        let root_block_fp = (SliceInput::new(&root_code.first).read_vlong()? as u64
            >> OUTPUT_FLAGS_NUM_BITS) as i64;
        let start = usize::try_from(index_start_fp)
            .ok()
            .filter(|&s| s <= tipv.len())
            .ok_or_else(|| corrupt(format!("invalid indexStartFP {index_start_fp}")))?;
        let (index, _) = TypedFst::<IndexOutputs>::read_prefix(&tipv[start..])
            .map_err(|e| corrupt(e.to_string()))?;
        if index.empty_output().is_none() {
            // Java asserts the first arc is final: the root prefix is empty.
            return Err(corrupt("terms index has no root output"));
        }
        Ok(VersionFieldReader {
            field_number,
            num_terms,
            root_code,
            root_block_fp,
            index_start_fp,
            min_term,
            max_term,
            index,
        })
    }

    /// `fieldInfo.number`.
    pub fn field_number(&self) -> i32 {
        self.field_number
    }

    /// `size()`: the number of terms.
    pub fn size(&self) -> i64 {
        self.num_terms
    }

    /// `getSumTotalTermFreq()`: one occurrence per term.
    pub fn sum_total_term_freq(&self) -> i64 {
        self.num_terms
    }

    /// `getSumDocFreq()`: one document per term.
    pub fn sum_doc_freq(&self) -> i64 {
        self.num_terms
    }

    /// `getDocCount()`.
    pub fn doc_count(&self) -> i32 {
        // ≤ maxDoc, checked on open.
        self.num_terms as i32
    }

    /// `getMin()`.
    pub fn min(&self) -> &[u8] {
        &self.min_term
    }

    /// `getMax()`.
    pub fn max(&self) -> &[u8] {
        &self.max_term
    }

    /// `rootCode`.
    pub fn root_code(&self) -> &IndexOutput {
        &self.root_code
    }

    /// `rootBlockFP`.
    pub fn root_block_fp(&self) -> i64 {
        self.root_block_fp
    }

    /// `indexStartFP`.
    pub fn index_start_fp(&self) -> i64 {
        self.index_start_fp
    }
}
