//! `org.apache.lucene.search.suggest.document`'s completion postings format at
//! the file level (M13 T13.3d): the weighted FST a suggest field is indexed
//! into (`NRTSuggesterBuilder`), the two files that hold it
//! (`CompletionFieldsConsumer`: `.lkp`, the FSTs; `.cmp`, the field table),
//! and the reader that opens them (`CompletionFieldsProducer`,
//! `CompletionsTermsReader`, `NRTSuggester.load`).
//!
//! A suggest field is an ordinary indexed field whose single token per value
//! carries a payload (`vint len, surface form, vint weight + 1, byte type`);
//! the delegate postings format indexes it as it would any field, and this
//! layer additionally walks every term's postings and builds one FST per field
//! from `analyzed form + END_BYTE + dedup bytes` to `(encoded weight, surface
//! form + SEP + doc id)`.
//!
//! The functions here take and return bytes, not directories: the caller
//! (the index writer or reader) owns the files.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::{DataInput, DataOutput, Error, Result, SliceInput};

use crate::fst::{ByteSequenceOutputs, InputType, Pair, PairOutputs, PositiveIntOutputs};
use crate::fst_compiler::{FstCompiler, FstCompilerBuilder};
use crate::fst_util::{to_ints_ref, TypedFst};

/// `CompletionPostingsFormat.COMPLETION_CODEC_VERSION` (and `_CURRENT`).
pub const COMPLETION_CODEC_VERSION: i32 = 1;
/// `CompletionPostingsFormat.INDEX_EXTENSION`.
pub const INDEX_EXTENSION: &str = "cmp";
/// `CompletionPostingsFormat.DICT_EXTENSION`.
pub const DICT_EXTENSION: &str = "lkp";
/// `NRTSuggesterBuilder.PAYLOAD_SEP`: `ConcatenateGraphFilter.SEP_LABEL`.
pub const PAYLOAD_SEP: u8 = 0x1F;
/// `NRTSuggesterBuilder.END_BYTE`.
pub const END_BYTE: u8 = 0x00;

/// The FST's outputs: `PairOutputs<Long, BytesRef>`.
pub type NrtOutputs = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;

fn illegal(message: impl Into<String>) -> Error {
    Error::IllegalArgument(message.into())
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::Corrupted(message.into())
}

/// `NRTSuggester.encode`: weights are stored as `Integer.MAX_VALUE - weight`
/// so the heaviest has the smallest output.
pub fn encode_weight(weight: i64) -> Result<i64> {
    if !(0..=i64::from(i32::MAX)).contains(&weight) {
        return Err(illegal(format!("cannot encode value: {weight}")));
    }
    // checked above: `weight` is in `0..=i32::MAX`
    Ok(i64::from(i32::MAX).wrapping_sub(weight))
}

/// `NRTSuggester.decode`.
/// Java's `long` subtraction, which wraps on an output read off a corrupt file.
pub fn decode_weight(output: i64) -> i64 {
    i64::from(i32::MAX).wrapping_sub(output)
}

/// `NRTSuggester.PayLoadProcessor.make`: `surface, SEP, vint docID`.
pub fn make_payload(surface: &[u8], doc_id: i32) -> Vec<u8> {
    let mut out = surface.to_vec();
    out.push(PAYLOAD_SEP);
    out.write_vint(doc_id);
    out
}

/// The index of the first `sep` in an output's second half: where the surface
/// form ends and the doc id begins.
pub fn payload_sep_index(output: &[u8], sep: u8) -> Option<usize> {
    output.iter().position(|&b| b == sep)
}

/// `NRTSuggesterBuilder.maxNumArcsForDedupByte`.
// ARITH: an i32 widened to i64: `2x + 1` and `x * x` cannot overflow i64.
#[allow(clippy::arithmetic_side_effects)]
fn max_num_arcs_for_dedup_byte(current_num_dedup_bytes: i32) -> i32 {
    let mut max_arcs = 2 * i64::from(current_num_dedup_bytes) + 1;
    if max_arcs >= 255 {
        return 255;
    }
    if current_num_dedup_bytes > 5 {
        max_arcs *= i64::from(current_num_dedup_bytes);
    }
    max_arcs.min(255) as i32
}

/// One surface form of a term, waiting in the builder's queue.
#[derive(Debug)]
struct Entry {
    payload: Vec<u8>,
    weight: i64,
}

/// `NRTSuggesterBuilder`.
pub struct NrtSuggesterBuilder {
    compiler: FstCompiler<NrtOutputs>,
    /// `java.util.PriorityQueue<Entry>` ordered by the (encoded) weight. Only
    /// its array order matters, since `finishTerm` iterates it as a
    /// collection; [`offer`](Self::offer) keeps Java's sift-up so that order
    /// is Java's.
    entries: Vec<Entry>,
    analyzed: Vec<u8>,
    max_analyzed_paths_per_output: i32,
}

impl std::fmt::Debug for NrtSuggesterBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NrtSuggesterBuilder")
            .field("pending", &self.entries.len())
            .field(
                "max_analyzed_paths_per_output",
                &self.max_analyzed_paths_per_output,
            )
            .finish()
    }
}

impl Default for NrtSuggesterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NrtSuggesterBuilder {
    /// `new NRTSuggesterBuilder()`.
    pub fn new() -> Self {
        Self {
            compiler: FstCompilerBuilder::new(InputType::Byte1).build::<NrtOutputs>(),
            entries: Vec::new(),
            analyzed: Vec::new(),
            max_analyzed_paths_per_output: 0,
        }
    }

    /// `startTerm(analyzed)`.
    pub fn start_term(&mut self, analyzed: &[u8]) {
        self.analyzed.clear();
        self.analyzed.extend_from_slice(analyzed);
        self.analyzed.push(END_BYTE);
    }

    /// `addEntry(docID, surfaceForm, weight)`.
    pub fn add_entry(&mut self, doc_id: i32, surface_form: &[u8], weight: i64) -> Result<()> {
        let payload = make_payload(surface_form, doc_id);
        self.offer(Entry {
            payload,
            weight: encode_weight(weight)?,
        });
        Ok(())
    }

    /// `PriorityQueue.offer`: append, then sift up while strictly smaller
    /// than the parent.
    // ARITH: `len() >= 1` right after the push, and `k > 0` in the loop.
    #[allow(clippy::arithmetic_side_effects)]
    fn offer(&mut self, entry: Entry) {
        self.entries.push(entry);
        let mut k = self.entries.len() - 1;
        while k > 0 {
            let parent = (k - 1) >> 1;
            if self.entries[k].weight >= self.entries[parent].weight {
                break;
            }
            self.entries.swap(k, parent);
            k = parent;
        }
    }

    /// `finishTerm()`: adds one FST path per queued surface form, told apart
    /// by dedup bytes appended to the analyzed form.
    // ARITH: `num_arcs` resets once it reaches `max_num_arcs_for_dedup_byte`
    // (at most 255), and `num_dedup_bytes` grows once per 3+ entries, so
    // both stay far below i32::MAX for any in-memory entry list.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn finish_term(&mut self) -> Result<()> {
        let mut num_arcs = 0i32;
        let mut num_dedup_bytes = 1i32;
        self.analyzed.push(0);
        let entries = std::mem::take(&mut self.entries);
        for entry in &entries {
            if num_arcs == max_num_arcs_for_dedup_byte(num_dedup_bytes) {
                if let Some(last) = self.analyzed.last_mut() {
                    *last = num_arcs as u8;
                }
                self.analyzed.push(0);
                num_arcs = 0;
                num_dedup_bytes += 1;
            }
            if let Some(last) = self.analyzed.last_mut() {
                *last = num_arcs as u8;
            }
            num_arcs += 1;
            self.compiler
                .add(
                    &to_ints_ref(&self.analyzed),
                    Pair {
                        first: entry.weight,
                        second: entry.payload.clone(),
                    },
                )
                .map_err(|e| illegal(e.to_string()))?;
        }
        let paths = i32::try_from(entries.len()).unwrap_or(i32::MAX);
        self.max_analyzed_paths_per_output = self.max_analyzed_paths_per_output.max(paths);
        Ok(())
    }

    /// `store(output)`: the FST and the three trailing vints; `false` (and
    /// nothing written) when no term ever added a path.
    pub fn store(&mut self, output: &mut Vec<u8>) -> Result<bool> {
        let Some(fst) = self.compiler.compile() else {
            return Ok(false);
        };
        output.extend_from_slice(&fst.save::<NrtOutputs>());
        output.write_vint(self.max_analyzed_paths_per_output);
        output.write_vint(i32::from(END_BYTE));
        output.write_vint(i32::from(PAYLOAD_SEP));
        Ok(true)
    }
}

/// A term's documents and, per document, the payload of each position.
#[derive(Debug, Clone)]
pub struct TermDoc {
    pub doc_id: i32,
    pub payloads: Vec<Vec<u8>>,
}

/// `CompletionFieldsConsumer.CompletionTermWriter`: one field's terms, in
/// term order, into a [`NrtSuggesterBuilder`].
#[derive(Debug)]
pub struct CompletionFieldWriter {
    builder: NrtSuggesterBuilder,
    doc_count: i32,
    pub max_weight: i64,
    pub min_weight: i64,
    pub type_byte: u8,
    first: bool,
}

impl Default for CompletionFieldWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionFieldWriter {
    /// `new CompletionTermWriter()`.
    pub fn new() -> Self {
        Self {
            builder: NrtSuggesterBuilder::new(),
            doc_count: 0,
            max_weight: 0,
            min_weight: i64::MAX,
            type_byte: 0,
            first: true,
        }
    }

    /// `write(term, termsEnum)`: the term's postings with payloads.
    pub fn write_term(&mut self, term: &[u8], docs: &[TermDoc]) -> Result<()> {
        self.builder.start_term(term);
        let mut doc_freq = 0i32;
        for doc in docs {
            for payload in &doc.payloads {
                let mut input = SliceInput::new(payload);
                let len = input.read_vint()?;
                let len = usize::try_from(len).map_err(|_| corrupt("negative surface length"))?;
                let mut surface = vec![0u8; len];
                input.read_bytes(&mut surface)?;
                let weight = i64::from(input.read_vint()?.wrapping_sub(1));
                self.max_weight = self.max_weight.max(weight);
                self.min_weight = self.min_weight.min(weight);
                let type_byte = input.read_byte()?;
                if self.first {
                    self.type_byte = type_byte;
                    self.first = false;
                } else if self.type_byte != type_byte {
                    return Err(illegal("single field name has mixed types"));
                }
                self.builder.add_entry(doc.doc_id, &surface, weight)?;
            }
            doc_freq = doc_freq.saturating_add(1);
            self.doc_count = self.doc_count.max(doc_freq.saturating_add(1));
        }
        self.builder.finish_term()
    }

    /// `finish(output)`: the field's FST block onto `output`; `false` when the
    /// field held no suggestions.
    pub fn finish(&mut self, output: &mut Vec<u8>) -> Result<bool> {
        let stored = self.builder.store(output)?;
        if self.doc_count == 0 {
            self.min_weight = 0;
        }
        Ok(stored)
    }
}

/// `CompletionFieldsConsumer.CompletionMetaData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionMetaData {
    pub file_pointer: u64,
    pub min_weight: i64,
    pub max_weight: i64,
    pub type_byte: u8,
}

/// The `.lkp` and `.cmp` of one segment's completion postings format.
#[derive(Debug)]
pub struct CompletionFilesWriter {
    codec_name: String,
    segment_id: [u8; ID_LENGTH],
    suffix: String,
    delegate_format: String,
    dict: Vec<u8>,
    seen: Vec<(String, i32, CompletionMetaData)>,
}

impl CompletionFilesWriter {
    /// The consumer's constructor: opens `.lkp` with its index header.
    pub fn new(
        codec_name: &str,
        delegate_format: &str,
        segment_id: [u8; ID_LENGTH],
        suffix: &str,
    ) -> Self {
        let mut dict = Vec::new();
        codec_util::write_index_header(
            &mut dict,
            codec_name,
            COMPLETION_CODEC_VERSION,
            &segment_id,
            suffix,
        );
        Self {
            codec_name: codec_name.to_string(),
            segment_id,
            suffix: suffix.to_string(),
            delegate_format: delegate_format.to_string(),
            dict,
            seen: Vec::new(),
        }
    }

    /// One field of `write(fields, norms)`, after the delegate wrote it. A
    /// field with no suggestions leaves no entry. Call it in field-name order,
    /// the order Java's `Fields` iterates: fields that share a `HashMap`
    /// bucket are written to `.cmp` in the order they were added.
    pub fn add_field(
        &mut self,
        name: &str,
        number: i32,
        mut writer: CompletionFieldWriter,
    ) -> Result<()> {
        let file_pointer = self.dict.len() as u64;
        if writer.finish(&mut self.dict)? {
            self.seen.push((
                name.to_string(),
                number,
                CompletionMetaData {
                    file_pointer,
                    min_weight: writer.min_weight,
                    max_weight: writer.max_weight,
                    type_byte: writer.type_byte,
                },
            ));
        }
        Ok(())
    }

    /// `close()`: both files, `.lkp` then `.cmp`, footers included. The field
    /// table is in `HashMap` order, as Java writes it.
    pub fn finish(self) -> (Vec<u8>, Vec<u8>) {
        let mut cmp = Vec::new();
        codec_util::write_index_header(
            &mut cmp,
            &self.codec_name,
            COMPLETION_CODEC_VERSION,
            &self.segment_id,
            &self.suffix,
        );
        cmp.write_string(&self.delegate_format);
        cmp.write_vint(i32::try_from(self.seen.len()).unwrap_or(i32::MAX));
        let names: Vec<&str> = self.seen.iter().map(|(n, _, _)| n.as_str()).collect();
        for i in java_hash_map_order(&names) {
            let (_, number, meta) = &self.seen[i];
            cmp.write_vint(*number);
            cmp.write_vlong(i64::try_from(meta.file_pointer).unwrap_or(i64::MAX));
            cmp.write_vlong(meta.min_weight);
            cmp.write_vlong(meta.max_weight);
            cmp.write_byte(meta.type_byte);
        }
        codec_util::write_footer(&mut cmp);
        let mut dict = self.dict;
        codec_util::write_footer(&mut dict);
        (dict, cmp)
    }
}

/// The order `java.util.HashMap<String, _>` iterates `keys` after they were
/// put in this order: by bucket (`hash ^ hash >>> 16` masked to the table
/// size), then by insertion, since a resize splits each chain keeping its
/// order. The table size is replayed put by put: it doubles once the size
/// passes three quarters of it, and also when a put appends a ninth key to one
/// bucket while the table is below 64 slots (`treeifyBin` resizes instead of
/// treeifying). Not reproduced: the order inside a treeified bucket (nine or
/// more colliding keys in a table of 64 or more), which follows the red-black
/// tree rather than insertion; no field table comes near it.
// ARITH: `capacity` doubles at most twice per put and never passes four times
// the key count, which is far below `usize::MAX`; `capacity - 1 >= 15`.
#[allow(clippy::arithmetic_side_effects)]
pub fn java_hash_map_order(keys: &[&str]) -> Vec<usize> {
    const TREEIFY_THRESHOLD: usize = 8;
    const MIN_TREEIFY_CAPACITY: usize = 64;
    let hashes: Vec<u32> = keys
        .iter()
        .map(|k| {
            let h = k
                .encode_utf16()
                .fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
                as u32;
            h ^ (h >> 16)
        })
        .collect();
    let mut capacity = 16usize;
    for (size, &h) in hashes.iter().enumerate() {
        let bucket = h as usize & (capacity - 1);
        let chain = hashes[..size]
            .iter()
            .filter(|&&o| o as usize & (capacity - 1) == bucket)
            .count();
        if chain >= TREEIFY_THRESHOLD && capacity < MIN_TREEIFY_CAPACITY {
            capacity *= 2;
        }
        if size + 1 > capacity / 4 * 3 {
            capacity *= 2;
        }
    }
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&i| hashes[i] as usize & (capacity - 1));
    order
}

/// `CompletionsTermsReader`: a field's weight range, type and the offset of
/// its FST block in `.lkp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionsTermsReader {
    pub min_weight: i64,
    pub max_weight: i64,
    pub type_byte: u8,
    pub offset: usize,
}

/// `NRTSuggester`'s loaded state: the FST and the three numbers after it.
#[derive(Debug, Clone)]
pub struct NrtSuggester {
    pub fst: TypedFst<NrtOutputs>,
    pub max_analyzed_paths_per_output: i32,
    pub payload_sep: i32,
}

impl NrtSuggester {
    /// `NRTSuggester.load(input)`, with `input` the `.lkp` from the field's
    /// offset.
    pub fn load(input: &[u8]) -> Result<Self> {
        let (fst, used) = TypedFst::<NrtOutputs>::read_prefix(input)
            .map_err(|e| corrupt(format!("completion FST: {e}")))?;
        let mut rest = SliceInput::new(&input[used..]);
        let max_analyzed_paths_per_output = rest.read_vint()?;
        if max_analyzed_paths_per_output < 1 {
            // the builder only stores an FST that has at least one path
            return Err(corrupt(format!(
                "maxAnalyzedPathsPerOutput {max_analyzed_paths_per_output}"
            )));
        }
        let _end_byte = rest.read_vint()?;
        let payload_sep = rest.read_vint()?;
        Ok(Self {
            fst,
            max_analyzed_paths_per_output,
            payload_sep,
        })
    }
}

/// `CompletionFieldsProducer`'s constructor: the field table of `.cmp` and
/// the bytes of `.lkp`.
#[derive(Debug)]
pub struct CompletionFields {
    /// The delegate postings format's name.
    pub delegate_format: String,
    /// Field number to its reader, in file order.
    pub fields: Vec<(i32, CompletionsTermsReader)>,
    dict: Vec<u8>,
    /// `CompletionsTermsReader.suggester`: each field's FST, loaded on first
    /// use and kept, as Java's lazily-set field is.
    loaded: Vec<std::sync::OnceLock<std::sync::Arc<NrtSuggester>>>,
}

impl CompletionFields {
    /// Opens both files: checks headers, ids and suffixes, the `.lkp` footer's
    /// shape and the `.cmp` checksum.
    pub fn open(
        codec_name: &str,
        segment_id: &[u8; ID_LENGTH],
        suffix: &str,
        cmp: &[u8],
        lkp: Vec<u8>,
    ) -> Result<Self> {
        let mut dict_in = SliceInput::new(&lkp);
        codec_util::check_index_header(
            &mut dict_in,
            codec_name,
            COMPLETION_CODEC_VERSION,
            COMPLETION_CODEC_VERSION,
            segment_id,
            suffix,
        )?;
        codec_util::retrieve_checksum(&lkp)?;

        let mut index = SliceInput::new(cmp);
        codec_util::check_index_header(
            &mut index,
            codec_name,
            COMPLETION_CODEC_VERSION,
            COMPLETION_CODEC_VERSION,
            segment_id,
            suffix,
        )?;
        let delegate_format = index.read_string()?;
        let num_fields = index.read_vint()?;
        let mut fields = Vec::new();
        for _ in 0..num_fields {
            let number = index.read_vint()?;
            let offset = index.read_vlong()?;
            let min_weight = index.read_vlong()?;
            let max_weight = index.read_vlong()?;
            let type_byte = index.read_byte()?;
            let offset = usize::try_from(offset)
                .ok()
                .filter(|&o| o < lkp.len())
                .ok_or_else(|| {
                    corrupt(format!("completion offset {offset} past the dictionary"))
                })?;
            fields.push((
                number,
                CompletionsTermsReader {
                    min_weight,
                    max_weight,
                    type_byte,
                    offset,
                },
            ));
        }
        codec_util::check_footer(&mut index, cmp.len())?;
        let loaded = fields.iter().map(|_| std::sync::OnceLock::new()).collect();
        Ok(Self {
            delegate_format,
            fields,
            dict: lkp,
            loaded,
        })
    }

    /// `CompletionsTermsReader.suggester()` for a field number.
    /// Loaded once per field; a load that fails is not kept.
    pub fn suggester(&self, field_number: i32) -> Result<Option<std::sync::Arc<NrtSuggester>>> {
        let Some(i) = self.fields.iter().position(|(n, _)| *n == field_number) else {
            return Ok(None);
        };
        let (Some((_, reader)), Some(slot)) = (self.fields.get(i), self.loaded.get(i)) else {
            return Ok(None);
        };
        if let Some(loaded) = slot.get() {
            return Ok(Some(std::sync::Arc::clone(loaded)));
        }
        let loaded = std::sync::Arc::new(NrtSuggester::load(&self.dict[reader.offset..])?);
        Ok(Some(std::sync::Arc::clone(slot.get_or_init(|| loaded))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fst_util::{get_bytes, TypedFst};

    fn payload(surface: &str, weight: i32, type_byte: u8) -> Vec<u8> {
        let mut out = Vec::new();
        out.write_vint(surface.len() as i32);
        out.extend_from_slice(surface.as_bytes());
        out.write_vint(weight.wrapping_add(1));
        out.push(type_byte);
        out
    }

    fn doc(id: i32, surface: &str, weight: i32) -> TermDoc {
        TermDoc {
            doc_id: id,
            payloads: vec![payload(surface, weight, 0)],
        }
    }

    #[test]
    fn weights_encode_to_the_complement_and_back() {
        assert_eq!(encode_weight(0).unwrap(), i64::from(i32::MAX));
        assert_eq!(encode_weight(i64::from(i32::MAX)).unwrap(), 0);
        assert_eq!(decode_weight(encode_weight(1234).unwrap()), 1234);
        assert!(encode_weight(-1).is_err());
        assert!(encode_weight(i64::from(i32::MAX) + 1).is_err());
    }

    #[test]
    fn payload_is_surface_sep_and_vint_doc() {
        let p = make_payload(b"caf\xc3\xa9", 300);
        assert_eq!(&p[..5], b"caf\xc3\xa9");
        assert_eq!(p[5], PAYLOAD_SEP);
        assert_eq!(&p[6..], &[0xAC, 0x02]);
        assert_eq!(payload_sep_index(&p, PAYLOAD_SEP), Some(5));
        assert_eq!(payload_sep_index(b"none", PAYLOAD_SEP), None);
    }

    #[test]
    fn dedup_bytes_grow_with_the_number_of_dedup_bytes() {
        assert_eq!(max_num_arcs_for_dedup_byte(1), 3);
        assert_eq!(max_num_arcs_for_dedup_byte(5), 11);
        assert_eq!(max_num_arcs_for_dedup_byte(6), 13 * 6);
        assert_eq!(max_num_arcs_for_dedup_byte(7), 15 * 7);
        assert_eq!(max_num_arcs_for_dedup_byte(200), 255);
        assert_eq!(max_num_arcs_for_dedup_byte(20), 255);
    }

    #[test]
    fn hash_map_order_is_by_bucket_then_insertion() {
        // "a" = 97 -> bucket 1, "b" -> 2, "p" = 112 -> 0
        assert_eq!(java_hash_map_order(&["b", "a", "p"]), vec![2, 1, 0]);
        assert_eq!(java_hash_map_order(&[]), Vec::<usize>::new());
        // "q" = 113 shares bucket 1 with "a" in 16 slots but not in 32: the
        // thirteenth key resizes the table, so "q" moves behind "l"
        let mut keys = vec!["q".to_string()];
        keys.extend((b'a'..=b'l').map(|c| (c as char).to_string()));
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let want: Vec<usize> = (1..13).chain([0]).collect();
        assert_eq!(java_hash_map_order(&refs), want);
        assert_eq!(java_hash_map_order(&refs[..12])[..2], [0, 1]);
    }

    #[test]
    fn a_field_round_trips_through_both_files() {
        let id = [7u8; ID_LENGTH];
        let mut w = CompletionFieldWriter::new();
        w.write_term(b"cat", &[doc(0, "Cat", 5), doc(3, "cat", 9)])
            .unwrap();
        w.write_term(b"dog", &[doc(1, "Dog", 2)]).unwrap();
        let mut files =
            CompletionFilesWriter::new("Completion104", "Lucene104", id, "Completion104_0");
        files.add_field("suggest", 4, w).unwrap();
        // a field that held nothing leaves no entry
        files
            .add_field("empty", 5, CompletionFieldWriter::new())
            .unwrap();
        let (lkp, cmp) = files.finish();

        let opened =
            CompletionFields::open("Completion104", &id, "Completion104_0", &cmp, lkp).unwrap();
        assert_eq!(opened.delegate_format, "Lucene104");
        assert_eq!(opened.fields.len(), 1);
        let (number, meta) = opened.fields[0];
        assert_eq!(number, 4);
        assert_eq!(
            (meta.min_weight, meta.max_weight, meta.type_byte),
            (2, 9, 0)
        );
        assert!(opened.suggester(5).unwrap().is_none());
        let s = opened.suggester(4).unwrap().unwrap();
        assert_eq!(s.payload_sep, i32::from(PAYLOAD_SEP));
        assert_eq!(s.max_analyzed_paths_per_output, 2);
        // "dog" + END_BYTE + dedup byte 0
        let out = get_bytes::<NrtOutputs>(&s.fst, b"dog\x00\x00")
            .unwrap()
            .unwrap();
        assert_eq!(decode_weight(out.first), 2);
        assert_eq!(out.second, make_payload(b"Dog", 1));
        let _: &TypedFst<NrtOutputs> = &s.fst;
    }

    #[test]
    fn open_refuses_a_wrong_segment_or_a_damaged_table() {
        let id = [1u8; ID_LENGTH];
        let mut w = CompletionFieldWriter::new();
        w.write_term(b"a", &[doc(0, "a", 1)]).unwrap();
        let mut files = CompletionFilesWriter::new("Completion104", "Lucene104", id, "S");
        files.add_field("f", 0, w).unwrap();
        let (lkp, cmp) = files.finish();
        let other = [2u8; ID_LENGTH];
        assert!(CompletionFields::open("Completion104", &other, "S", &cmp, lkp.clone()).is_err());
        assert!(CompletionFields::open("Completion104", &id, "T", &cmp, lkp.clone()).is_err());
        assert!(CompletionFields::open("Other", &id, "S", &cmp, lkp.clone()).is_err());
        let mut bad = cmp.clone();
        let n = bad.len();
        bad[n - 10] ^= 0xFF;
        assert!(CompletionFields::open("Completion104", &id, "S", &bad, lkp.clone()).is_err());
        assert!(
            CompletionFields::open("Completion104", &id, "S", &cmp, lkp[..8].to_vec()).is_err()
        );
    }

    #[test]
    fn mixed_types_and_bad_weights_are_refused() {
        let mut w = CompletionFieldWriter::new();
        w.write_term(
            b"a",
            &[TermDoc {
                doc_id: 0,
                payloads: vec![payload("a", 1, 0), payload("a", 2, 1)],
            }],
        )
        .unwrap_err();
        // a surface form whose weight does not fit
        let mut w = CompletionFieldWriter::new();
        let bad = TermDoc {
            doc_id: 0,
            payloads: vec![{
                let mut p = Vec::new();
                p.write_vint(1);
                p.push(b'a');
                p.write_vint(0); // weight -1
                p.push(0);
                p
            }],
        };
        assert!(w.write_term(b"a", &[bad]).is_err());
    }

    #[test]
    fn load_refuses_a_non_positive_path_count() {
        let mut w = CompletionFieldWriter::new();
        w.write_term(b"a", &[doc(0, "a", 1)]).unwrap();
        let mut out = Vec::new();
        assert!(w.finish(&mut out).unwrap());
        // the three trailing vints: paths per output, END_BYTE, PAYLOAD_SEP
        let n = out.len();
        out[n - 3] = 0;
        assert!(NrtSuggester::load(&out).is_err());
    }

    #[test]
    fn many_surface_forms_of_one_term_get_dedup_bytes() {
        let mut w = CompletionFieldWriter::new();
        let docs: Vec<TermDoc> = (0..40).map(|i| doc(i, "x", i)).collect();
        w.write_term(b"x", &docs).unwrap();
        let mut out = Vec::new();
        assert!(w.finish(&mut out).unwrap());
        let s = NrtSuggester::load(&out).unwrap();
        assert_eq!(s.max_analyzed_paths_per_output, 40);
    }
}
