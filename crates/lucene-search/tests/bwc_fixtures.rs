//! M8 T8.2/T8.3's differential harness: every `fixtures/data/bwc/<version>/`
//! index, written by that Lucene release's own jars (`fixtures/bwc/BwcWrite.java`),
//! read back by this port and compared line by line against the
//! `expected.txt` that Lucene 10.5.0 with backward-codecs wrote for it
//! (`fixtures/bwc/BwcDump.java`).
//!
//! Every line kind `BwcDump` writes is reproduced here from this port's own
//! readers, over the exact canonical byte streams its class javadoc defines
//! (and, where the javadoc and the code disagree -- `dv` SORTED/SORTED_SET
//! start with `L(valueCount)` -- the code, which is what produced the
//! digests):
//!
//! - `commit`, `seg`, `field`: the commit, `SegmentCommitInfo` and `FieldInfo`
//!   state, in full.
//! - `live`, `postings` (with its statistics), `norms`, `dv`, `points` (with
//!   size/docCount/min/max), `stored`, `tv`, `vec`: FNV-1a 64 digests.
//! - `knn`: the ten nearest live documents with their score bits.
//!
//! Each `(version, line)` is checked on its own -- a segment whose postings
//! do not open still has its norms, doc values and stored fields compared --
//! and the report lists every mismatch. [`EXPECTED_FAILURES`] names the lines
//! this port cannot read yet; it must only shrink: a listed line that starts
//! passing fails the test until it is removed, so the table stays honest.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lucene_codecs::backward_codecs::hnsw_vectors::{RetiredHnswFormat, RetiredHnswVectorsReader};
use lucene_codecs::blocktree;
use lucene_codecs::doc_values::{self, SortedSetKind};
use lucene_codecs::field_infos::{
    self, DocValuesType, FieldInfo, FieldInfos, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
};
use lucene_codecs::live_docs;
use lucene_codecs::norms;
use lucene_codecs::points::{self, IntersectVisitor, Relation};
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_codecs::stored_fields::{self, StoredFieldVisitor, VisitStatus};
use lucene_codecs::term_vectors;
use lucene_codecs::terms_dict::TermsDict;
use lucene_codecs::vectors::{FlatVectorsReader, MergeSourceValues};
use lucene_index::deletes::liv_file_name;
use lucene_index::segment_info::{self, SegmentInfo};
use lucene_index::segment_infos::{self, SegmentCommitInfo};
use lucene_search::vector_query::GraphReader;
use lucene_store::directory::{Directory, FsDirectory, Input};
use lucene_util::fixed_bit_set::FixedBitSet;

/// Every fixture version `scripts/gen-bwc-fixtures.sh` writes, oldest first.
const VERSIONS: &[&str] = &[
    "9.0.0", "9.1.0", "9.3.0", "9.4.2", "9.8.0", "9.11.1", "9.12.2", "10.0.0", "10.2.2", "10.4.0",
];

/// Lines this port does not reproduce yet: `(version, kind, field)`, where
/// `field` is the field name for per-field lines and `*` matches every field
/// (or the segment-level line) of that kind in both segments.
///
/// Empty since the retired `Lucene90`..`Lucene95` HNSW readers landed (the
/// last `vec`/`knn` lines of 9.0-9.8); kept so a future fixture version can
/// record a known gap without weakening the whole-file comparison.
const EXPECTED_FAILURES: &[(&str, &str, &str)] = &[];

/// `fixtures/data/bwc-quantized/<version>/` (`fixtures/bwc/BwcQuantized.java`):
/// per-field `Lucene99(Hnsw)ScalarQuantizedVectorsFormat` fields from every
/// release that wrote them at a distinct format version, and the
/// `Lucene102(Hnsw)BinaryQuantizedVectorsFormat` fields of 10.2.
const QUANTIZED_VERSIONS: &[&str] = &["9.9.2", "9.12.2", "10.2.2"];

fn fixture_dir(version: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/bwc")
        .join(version)
}

fn quantized_fixture_dir(version: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/bwc-quantized")
        .join(version)
}

// ---------------------------------------------------------------------------
// BwcDump's canonical stream.

/// `BwcDump.Fnv`: FNV-1a 64 over `L(x)` (8 little-endian bytes) and `B(b)`
/// (`L(len)` then the bytes).
struct Fnv {
    h: u64,
}

impl Fnv {
    fn new() -> Self {
        Fnv {
            h: 0xcbf2_9ce4_8422_2325,
        }
    }

    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.h ^= u64::from(x);
            self.h = self.h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn l(&mut self, v: i64) {
        self.bytes(&v.to_le_bytes());
    }

    fn b(&mut self, v: &[u8]) {
        self.l(v.len() as i64);
        self.bytes(v);
    }

    fn hex(&self) -> String {
        format!("{:016x}", self.h)
    }
}

// ---------------------------------------------------------------------------
// Java's `toString`s.

fn index_options_name(o: IndexOptions) -> &'static str {
    match o {
        IndexOptions::None => "NONE",
        IndexOptions::Docs => "DOCS",
        IndexOptions::DocsAndFreqs => "DOCS_AND_FREQS",
        IndexOptions::DocsAndFreqsAndPositions => "DOCS_AND_FREQS_AND_POSITIONS",
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => {
            "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS"
        }
        IndexOptions::DocsAndCustomFreqs => "DOCS_AND_CUSTOM_FREQS",
    }
}

fn dv_type_name(t: DocValuesType) -> &'static str {
    match t {
        DocValuesType::None => "NONE",
        DocValuesType::Numeric => "NUMERIC",
        DocValuesType::Binary => "BINARY",
        DocValuesType::Sorted => "SORTED",
        DocValuesType::SortedSet => "SORTED_SET",
        DocValuesType::SortedNumeric => "SORTED_NUMERIC",
    }
}

fn encoding_name(e: VectorEncoding) -> &'static str {
    match e {
        VectorEncoding::Byte => "BYTE",
        VectorEncoding::Float32 => "FLOAT32",
    }
}

fn similarity_name(s: VectorSimilarityFunction) -> &'static str {
    match s {
        VectorSimilarityFunction::Euclidean => "EUCLIDEAN",
        VectorSimilarityFunction::DotProduct => "DOT_PRODUCT",
        VectorSimilarityFunction::Cosine => "COSINE",
        VectorSimilarityFunction::MaximumInnerProduct => "MAXIMUM_INNER_PRODUCT",
    }
}

fn version_string(v: &segment_info::LuceneVersion) -> String {
    format!("{}.{}.{}", v.major, v.minor, v.bugfix)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn has_norms(fi: &FieldInfo) -> bool {
    fi.index_options != IndexOptions::None && !fi.omit_norms
}

// ---------------------------------------------------------------------------
// The harness.

type Lines = BTreeMap<String, Result<String, String>>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// `segment name` -> the files this segment's `SegmentCommitInfo.files()`
/// reports, plus everything opened from them once.
struct Segment {
    commit: SegmentCommitInfo,
    si: SegmentInfo,
    field_infos: FieldInfos,
    live: Option<FixedBitSet>,
}

impl Segment {
    fn name(&self) -> &str {
        &self.commit.segment_name
    }

    fn max_doc(&self) -> i32 {
        self.si.doc_count
    }

    fn file_with(&self, ext: &str) -> Option<&str> {
        self.si
            .files
            .iter()
            .find(|f| f.ends_with(ext))
            .map(String::as_str)
    }

    /// The per-format codec suffix embedded in a file name:
    /// `_0_Lucene90_0.dvm` -> `Lucene90_0`, `_0.fdt` -> ``.
    fn suffix_of(&self, file: &str, ext: &str) -> String {
        let stem = file.strip_suffix(ext).unwrap_or(file);
        stem.strip_prefix(&format!("{}_", self.name()))
            .unwrap_or("")
            .to_string()
    }

    fn is_live(&self, doc: i32) -> bool {
        self.live.as_ref().is_none_or(|l| l.get(doc as usize))
    }
}

fn open_segment(dir: &dyn Directory, commit: &SegmentCommitInfo) -> Result<Segment, String> {
    let si_bytes = dir
        .open(&format!("{}.si", commit.segment_name))
        .map_err(err)?;
    let si = segment_info::parse_for_codec(&si_bytes, &commit.segment_id, &commit.codec_name)
        .map_err(err)?;
    let fnm_name = si
        .files
        .iter()
        .find(|f| f.ends_with(".fnm"))
        .ok_or("no .fnm")?;
    let fnm = dir.open(fnm_name).map_err(err)?;
    let field_infos = field_infos::parse(&fnm, &commit.segment_id, "").map_err(err)?;
    let live = if commit.del_gen != -1 {
        let liv = dir
            .open(&liv_file_name(&commit.segment_name, commit.del_gen))
            .map_err(err)?;
        Some(
            live_docs::parse(
                &liv,
                &commit.segment_id,
                commit.del_gen,
                si.doc_count as usize,
                commit.del_count as usize,
            )
            .map_err(err)?,
        )
    } else {
        None
    };
    Ok(Segment {
        commit: commit.clone(),
        si,
        field_infos,
        live,
    })
}

fn seg_line(seg: &Segment) -> String {
    let c = &seg.commit;
    let mut files: Vec<String> = seg.si.files.clone();
    if c.del_gen != -1 {
        files.push(liv_file_name(&c.segment_name, c.del_gen));
    }
    files.extend(c.field_infos_files.iter().cloned());
    for (_, f) in &c.dv_update_files {
        files.extend(f.iter().cloned());
    }
    files.sort();
    files.dedup();
    format!(
        "seg {} codec={} maxDoc={} delCount={} softDel={} version={} minVersion={} compound={} files=[{}]",
        c.segment_name,
        c.codec_name,
        seg.si.doc_count,
        c.del_count,
        c.soft_del_count,
        version_string(&seg.si.version),
        seg.si
            .min_version
            .as_ref()
            .map_or("null".to_string(), version_string),
        seg.si.is_compound_file,
        files.join(", ")
    )
}

fn field_line(seg: &str, fi: &FieldInfo) -> String {
    let mut attrs: Vec<(String, String)> = fi.attributes.clone();
    attrs.sort();
    let attrs = attrs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "field {seg} {} num={} index={} tv={} norms={} payloads={} dv={} points={},{},{} vec={},{},{} attrs={{{attrs}}}",
        fi.name,
        fi.number,
        index_options_name(fi.index_options),
        fi.store_term_vectors,
        has_norms(fi),
        fi.store_payloads,
        dv_type_name(fi.doc_values_type),
        fi.point_dimension_count,
        fi.point_index_dimension_count,
        fi.point_num_bytes,
        fi.vector_dimension,
        encoding_name(fi.vector_encoding),
        similarity_name(fi.vector_similarity_function),
    )
}

fn live_line(seg: &Segment) -> String {
    let mut f = Fnv::new();
    let mut n = 0;
    for d in 0..seg.max_doc() {
        let alive = seg.is_live(d);
        f.l(i64::from(alive));
        if alive {
            n += 1;
        }
    }
    format!("live {} n={n} {}", seg.name(), f.hex())
}

/// The segment's postings files, opened once for every field.
struct PostingsFiles {
    tim: Input,
    tip: Input,
    tmd: Input,
    doc: Option<Input>,
    pos: Option<Input>,
    pay: Option<Input>,
    suffix: String,
}

fn open_postings_files(dir: &dyn Directory, seg: &Segment) -> Result<PostingsFiles, String> {
    let tim_name = seg.file_with(".tim").ok_or("no .tim")?;
    let open = |ext: &str| -> Result<Option<Input>, String> {
        seg.file_with(ext)
            .map(|n| dir.open(n).map_err(err))
            .transpose()
    };
    Ok(PostingsFiles {
        tim: dir.open(tim_name).map_err(err)?,
        tip: open(".tip")?.ok_or("no .tip")?,
        tmd: open(".tmd")?.ok_or("no .tmd")?,
        doc: open(".doc")?,
        pos: open(".pos")?,
        pay: open(".pay")?,
        suffix: seg.suffix_of(tim_name, ".tim"),
    })
}

fn postings_line(
    seg: &Segment,
    fi: &FieldInfo,
    fields: &blocktree::BlockTreeFields,
    files: &PostingsFiles,
) -> Result<String, String> {
    let id = &seg.commit.segment_id;
    let doc_in = files
        .doc
        .as_ref()
        .map(|b| DocInput::open(b, id, &files.suffix))
        .transpose()
        .map_err(err)?;
    let pos_in = files
        .pos
        .as_ref()
        .map(|b| PosInput::open(b, id, &files.suffix))
        .transpose()
        .map_err(err)?;
    let pay_in = files
        .pay
        .as_ref()
        .map(|b| PayInput::open(b, id, &files.suffix))
        .transpose()
        .map_err(err)?;
    let terms = fields
        .field(&fi.name)
        .ok_or_else(|| format!("no terms for {}", fi.name))?;
    let has_pos = fi.index_options.subsumes_positions();
    let has_off = fi.index_options.subsumes_offsets();
    let mut f = Fnv::new();
    let mut te = terms.iter();
    let mut term = Vec::new();
    while let Some((t, stats)) = te.try_next().map_err(err)? {
        term.clear();
        term.extend_from_slice(t);
        f.b(&term);
        f.l(i64::from(stats.doc_freq));
        f.l(stats.total_term_freq);
        if has_pos {
            let pos_in = pos_in.as_ref().ok_or("positions without .pos")?;
            let (docs, positions) = te
                .try_current_postings_and_positions(doc_in.as_ref(), pos_in, pay_in.as_ref())
                .map_err(err)?
                .ok_or("no postings on a term")?;
            for (i, &d) in docs.docs.iter().enumerate() {
                f.l(i64::from(d));
                let freq = docs.freqs[i];
                f.l(i64::from(freq));
                let occ = &positions[i];
                if occ.len() != freq as usize {
                    return Err(format!("doc {d}: {} positions for freq {freq}", occ.len()));
                }
                for p in occ {
                    f.l(i64::from(p.position));
                    if has_off {
                        f.l(i64::from(p.start_offset));
                        f.l(i64::from(p.end_offset));
                    }
                    if fi.store_payloads {
                        if p.payload.is_empty() {
                            f.l(-1);
                        } else {
                            f.b(&p.payload);
                        }
                    }
                }
            }
        } else {
            let docs = te
                .try_current_postings(doc_in.as_ref())
                .map_err(err)?
                .ok_or("no postings on a term")?;
            for (i, &d) in docs.docs.iter().enumerate() {
                f.l(i64::from(d));
                f.l(i64::from(docs.freqs[i]));
            }
        }
    }
    Ok(format!(
        "postings {} {} terms={} sumDocFreq={} sumTTF={} docCount={} min={} max={} {}",
        seg.name(),
        fi.name,
        terms.num_terms,
        terms.sum_doc_freq,
        terms.sum_total_term_freq,
        terms.doc_count,
        String::from_utf8_lossy(&terms.min_term),
        String::from_utf8_lossy(&terms.max_term),
        f.hex()
    ))
}

fn norms_line(
    seg: &Segment,
    fi: &FieldInfo,
    meta: &norms::Norms,
    data: &[u8],
) -> Result<String, String> {
    let entry = meta
        .entry(fi.number)
        .ok_or_else(|| format!("no norms entry for {}", fi.name))?;
    let mut f = Fnv::new();
    for d in 0..seg.max_doc() {
        if let Some(v) = norms::norm_value(data, entry, d).map_err(err)? {
            f.l(i64::from(d));
            f.l(v);
        }
    }
    Ok(format!("norms {} {} {}", seg.name(), fi.name, f.hex()))
}

fn term_of(dict: &mut TermsDict<'_>, ord: i64) -> Result<Vec<u8>, String> {
    Ok(dict.seek_ord(ord).map_err(err)?.to_vec())
}

fn dv_line(
    seg: &Segment,
    fi: &FieldInfo,
    meta: &doc_values::DocValuesMeta,
    dvd: &[u8],
) -> Result<String, String> {
    let n = fi.number;
    let missing = || format!("no .dvm entry for {}", fi.name);
    let mut f = Fnv::new();
    match fi.doc_values_type {
        DocValuesType::Numeric => {
            let e = meta.numeric_entry(n).ok_or_else(missing)?;
            for d in 0..seg.max_doc() {
                if let Some(v) = doc_values::numeric_value(dvd, e, d).map_err(err)? {
                    f.l(i64::from(d));
                    f.l(v);
                }
            }
        }
        DocValuesType::Binary => {
            let e = meta.binary_entry(n).ok_or_else(missing)?;
            for d in 0..seg.max_doc() {
                if let Some(v) = doc_values::binary_value(dvd, e, d).map_err(err)? {
                    f.l(i64::from(d));
                    f.b(v);
                }
            }
        }
        DocValuesType::Sorted => {
            let e = meta.sorted_entry(n).ok_or_else(missing)?;
            let mut dict = TermsDict::open(dvd, &e.terms).map_err(err)?;
            f.l(dict.size());
            for d in 0..seg.max_doc() {
                if let Some(ord) = doc_values::sorted_ord(dvd, e, d).map_err(err)? {
                    f.l(i64::from(d));
                    f.l(ord);
                    f.b(&term_of(&mut dict, ord)?);
                }
            }
        }
        DocValuesType::SortedSet => {
            let e = meta.sorted_set_entry(n).ok_or_else(missing)?;
            match &e.kind {
                SortedSetKind::Single(single) => {
                    let mut dict = TermsDict::open(dvd, &single.terms).map_err(err)?;
                    f.l(dict.size());
                    for d in 0..seg.max_doc() {
                        if let Some(ord) = doc_values::sorted_ord(dvd, single, d).map_err(err)? {
                            f.l(i64::from(d));
                            f.l(1);
                            f.l(ord);
                            f.b(&term_of(&mut dict, ord)?);
                        }
                    }
                }
                SortedSetKind::Multi { ords, terms } => {
                    let mut dict = TermsDict::open(dvd, terms).map_err(err)?;
                    f.l(dict.size());
                    for d in 0..seg.max_doc() {
                        let vals = doc_values::sorted_numeric_values(dvd, ords, d).map_err(err)?;
                        if !vals.is_empty() {
                            f.l(i64::from(d));
                            f.l(vals.len() as i64);
                            for ord in vals {
                                f.l(ord);
                                f.b(&term_of(&mut dict, ord)?);
                            }
                        }
                    }
                }
            }
        }
        DocValuesType::SortedNumeric => {
            let e = meta.sorted_numeric_entry(n).ok_or_else(missing)?;
            for d in 0..seg.max_doc() {
                let vals = doc_values::sorted_numeric_values(dvd, e, d).map_err(err)?;
                if !vals.is_empty() {
                    f.l(i64::from(d));
                    f.l(vals.len() as i64);
                    for v in vals {
                        f.l(v);
                    }
                }
            }
        }
        DocValuesType::None => return Err("no doc values".into()),
    }
    Ok(format!(
        "dv {} {} {} {}",
        seg.name(),
        fi.name,
        dv_type_name(fi.doc_values_type),
        f.hex()
    ))
}

/// `BwcDump`'s intersect visitor: every cell crosses, so every point is
/// visited with its value.
struct CollectAll {
    seen: Vec<(i32, Vec<u8>)>,
    bad: bool,
}

impl IntersectVisitor for CollectAll {
    fn compare(&mut self, _min: &[u8], _max: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _doc_id: i32) {
        self.bad = true;
    }
    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        self.seen.push((doc_id, packed_value.to_vec()));
    }
}

fn points_line(
    seg: &Segment,
    fi: &FieldInfo,
    reader: &points::PointsReader<'_>,
) -> Result<String, String> {
    let field = reader
        .field(fi.number)
        .ok_or_else(|| format!("no points for {}", fi.name))?;
    let mut v = CollectAll {
        seen: Vec::new(),
        bad: false,
    };
    reader.intersect(fi.number, &mut v).map_err(err)?;
    if v.bad {
        return Err("visit(docID) called on a crossing cell".into());
    }
    v.seen.sort();
    let mut f = Fnv::new();
    for (d, p) in &v.seen {
        f.l(i64::from(*d));
        f.b(p);
    }
    Ok(format!(
        "points {} {} size={} docCount={} min={} max={} {}",
        seg.name(),
        fi.name,
        field.point_count,
        field.doc_count,
        hex(&field.min_packed_value),
        hex(&field.max_packed_value),
        f.hex()
    ))
}

/// `BwcDump`'s stored-fields visitor: every field, straight into the digest.
struct StoredDigest<'a> {
    f: &'a mut Fnv,
    names: &'a FieldInfos,
}

impl StoredDigest<'_> {
    fn name(&mut self, n: i32) {
        let name = self
            .names
            .fields
            .iter()
            .find(|f| f.number == n)
            .map_or("", |f| f.name.as_str());
        self.f.b(name.as_bytes());
    }
}

impl StoredFieldVisitor for StoredDigest<'_> {
    fn needs_field(&mut self, _n: i32) -> stored_fields::Result<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn string_field(&mut self, n: i32, v: &str) -> stored_fields::Result<()> {
        self.name(n);
        self.f.l(0);
        self.f.b(v.as_bytes());
        Ok(())
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> stored_fields::Result<()> {
        self.name(n);
        self.f.l(1);
        self.f.b(v);
        Ok(())
    }
    fn int_field(&mut self, n: i32, v: i32) -> stored_fields::Result<()> {
        self.name(n);
        self.f.l(2);
        self.f.l(i64::from(v));
        Ok(())
    }
    fn long_field(&mut self, n: i32, v: i64) -> stored_fields::Result<()> {
        self.name(n);
        self.f.l(3);
        self.f.l(v);
        Ok(())
    }
    fn float_field(&mut self, n: i32, v: f32) -> stored_fields::Result<()> {
        self.name(n);
        self.f.l(4);
        self.f.l(i64::from(v.to_bits() as i32));
        Ok(())
    }
    fn double_field(&mut self, n: i32, v: f64) -> stored_fields::Result<()> {
        self.name(n);
        self.f.l(5);
        self.f.l(v.to_bits() as i64);
        Ok(())
    }
}

fn stored_line(dir: &dyn Directory, seg: &Segment) -> Result<String, String> {
    let fdt_name = seg.file_with(".fdt").ok_or("no .fdt")?;
    let fdt = dir.open(fdt_name).map_err(err)?;
    let fdx = dir
        .open(seg.file_with(".fdx").ok_or("no .fdx")?)
        .map_err(err)?;
    let fdm = dir
        .open(seg.file_with(".fdm").ok_or("no .fdm")?)
        .map_err(err)?;
    let reader = stored_fields::open(
        &fdt,
        &fdx,
        &fdm,
        &seg.commit.segment_id,
        &seg.suffix_of(fdt_name, ".fdt"),
    )
    .map_err(err)?;
    let mut f = Fnv::new();
    for d in 0..seg.max_doc() {
        f.l(i64::from(d));
        let mut v = StoredDigest {
            f: &mut f,
            names: &seg.field_infos,
        };
        reader.visit_document(d, &mut v).map_err(err)?;
    }
    Ok(format!("stored {} {}", seg.name(), f.hex()))
}

fn tv_line(dir: &dyn Directory, seg: &Segment) -> Result<String, String> {
    let Some(tvd_name) = seg.file_with(".tvd") else {
        // No term vectors at all: `L(-1)` for every document.
        let mut f = Fnv::new();
        for _ in 0..seg.max_doc() {
            f.l(-1);
        }
        return Ok(format!("tv {} {}", seg.name(), f.hex()));
    };
    let tvd = dir.open(tvd_name).map_err(err)?;
    let tvx = dir
        .open(seg.file_with(".tvx").ok_or("no .tvx")?)
        .map_err(err)?;
    let tvm = dir
        .open(seg.file_with(".tvm").ok_or("no .tvm")?)
        .map_err(err)?;
    let reader = term_vectors::open(
        &tvd,
        &tvx,
        &tvm,
        &seg.commit.segment_id,
        &seg.suffix_of(tvd_name, ".tvd"),
    )
    .map_err(err)?;
    let mut f = Fnv::new();
    for d in 0..seg.max_doc() {
        let Some(doc) = reader.document(d).map_err(err)? else {
            f.l(-1);
            continue;
        };
        for field in &doc.fields {
            let name = seg
                .field_infos
                .fields
                .iter()
                .find(|fi| fi.number == field.field_number)
                .map_or("", |fi| fi.name.as_str());
            f.b(name.as_bytes());
            let pos = field.has_positions || field.has_offsets;
            for t in &field.terms {
                f.b(&t.term);
                f.l(i64::from(t.freq));
                if !pos {
                    continue;
                }
                for k in 0..t.freq as usize {
                    let p = t.positions.as_ref().map_or(-1, |p| p[k]);
                    f.l(i64::from(p));
                    if field.has_offsets {
                        f.l(i64::from(t.start_offsets.as_ref().map_or(-1, |o| o[k])));
                        f.l(i64::from(t.end_offsets.as_ref().map_or(-1, |o| o[k])));
                    }
                    if field.has_payloads {
                        match t.payloads.as_ref().map(|p| &p[k]) {
                            Some(p) if !p.is_empty() => f.b(p),
                            _ => f.l(-1),
                        }
                    }
                }
            }
        }
    }
    Ok(format!("tv {} {}", seg.name(), f.hex()))
}

/// The per-field vector lines `BwcDump` writes for one field: `vec`, `knn`
/// and the query-level `knnf`, `knne`, `patience`, `vsim`, `vsimf`.
const VECTOR_LINE_KINDS: [&str; 7] = ["vec", "knn", "knnf", "knne", "patience", "vsim", "vsimf"];

/// `BwcDump.ModQuery`: the documents `doc % m == 0` of a segment.
fn mod_filter(max_doc: i32, m: i32) -> FixedBitSet {
    let mut bits = FixedBitSet::new(max_doc as usize);
    for d in (0..max_doc).step_by(m as usize) {
        bits.set(d as usize);
    }
    bits
}

fn hits_str(hits: &[lucene_search::ScoreDoc]) -> String {
    hits.iter()
        .map(|h| format!("{}:{:x}", h.doc_id, h.score.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

/// The files of one field's `PerFieldKnnVectorsFormat` group, by the
/// field's `format` and `suffix` attributes: `_0_<format>_<suffix>.<ext>`.
fn vector_files(
    dir: &dyn Directory,
    seg: &Segment,
    fi: &FieldInfo,
) -> Result<(String, String, BTreeMap<&'static str, Input>), String> {
    let attr = |key: &str| {
        fi.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .ok_or(format!("{} has no {key}", fi.name))
    };
    let format = attr("PerFieldKnnVectorsFormat.format")?;
    let suffix = format!("{format}_{}", attr("PerFieldKnnVectorsFormat.suffix")?);
    let mut files = BTreeMap::new();
    for ext in ["vec", "vem", "vex", "vemf", "vemq", "veq", "vemb", "veb"] {
        let name = format!("{}_{suffix}.{ext}", seg.name());
        if seg.si.files.contains(&name) {
            files.insert(ext, dir.open(&name).map_err(err)?);
        }
    }
    Ok((format, suffix, files))
}

fn vec_lines(dir: &dyn Directory, seg: &Segment, fi: &FieldInfo, out: &mut Lines) {
    use lucene_codecs::backward_codecs::quantized_vectors::{
        QuantizedFiles, QuantizedFormat, QuantizedVectorsReader,
    };
    use lucene_search::vector_query::{
        self as vq, ByteVectorSimilarityQuery, FloatVectorSimilarityQuery, KnnByteVectorQuery,
        KnnFloatVectorQuery, KnnSegment, PatienceKnnVectorQuery,
    };
    let keys: Vec<String> = VECTOR_LINE_KINDS
        .iter()
        .map(|k| format!("{k} {} {}", seg.name(), fi.name))
        .collect();
    let result = (|| -> Result<Vec<String>, String> {
        let (format, suffix, files) = vector_files(dir, seg, fi)?;
        let file = |ext: &str| -> Result<&[u8], String> {
            files
                .get(ext)
                .map(|i| &i[..])
                .ok_or(format!("{} has no .{ext}", fi.name))
        };
        let id = &seg.commit.segment_id;
        // `PerFieldKnnVectorsFormat.format`: a retired 9.0-9.8 format keeps
        // its vectors in the `.vem`/`.vec`/`.vex` triple; the current one
        // adds a `.vemf` for them; a quantized one adds its own pair too.
        let (flat, hnsw): (FlatVectorsReader<'_>, Option<GraphReader<'_>>) =
            if let Some(retired) = RetiredHnswFormat::for_name(&format) {
                let r = RetiredHnswVectorsReader::open(
                    retired,
                    file("vem")?,
                    file("vec")?,
                    file("vex")?,
                    id,
                    &suffix,
                )
                .map_err(err)?;
                (r.flat().clone(), Some(r.into()))
            } else if let Some(quantized) = QuantizedFormat::for_name(&format) {
                let (meta_ext, data_ext) = quantized.quantized_extensions();
                let graph = match (files.get("vem"), files.get("vex")) {
                    (Some(m), Some(x)) => Some((&m[..], &x[..])),
                    _ => None,
                };
                let r = QuantizedVectorsReader::open(
                    quantized,
                    QuantizedFiles {
                        vemf: file("vemf")?,
                        vec: file("vec")?,
                        quantized_meta: file(meta_ext)?,
                        quantized_data: file(data_ext)?,
                        graph,
                    },
                    &seg.field_infos,
                    id,
                    &suffix,
                )
                .map_err(err)?;
                r.check_integrity().map_err(err)?;
                (r.flat().clone(), Some(r.into()))
            } else {
                let flat = FlatVectorsReader::open(file("vemf")?, file("vec")?, id, &suffix)
                    .map_err(err)?;
                let graph = match (files.get("vem"), files.get("vex")) {
                    (Some(m), Some(x)) => Some(
                        lucene_codecs::hnsw_vectors::HnswVectorsReader::open(m, x, id, &suffix)
                            .map_err(err)?
                            .into(),
                    ),
                    _ => None,
                };
                (flat, graph)
            };
        let mut f = Fnv::new();
        let (values, count) = match fi.vector_encoding {
            VectorEncoding::Float32 => {
                let v = flat.float_vector_values(fi.number).map_err(err)?;
                let n = v.size();
                for ord in 0..n {
                    let src = MergeSourceValues::Float32(v.clone());
                    f.l(i64::from(src.ord_to_doc(ord).map_err(err)?));
                    for x in v.vector(ord).map_err(err)? {
                        f.l(i64::from(x.to_bits() as i32));
                    }
                }
                (MergeSourceValues::Float32(v), n)
            }
            VectorEncoding::Byte => {
                let v = flat.byte_vector_values(fi.number).map_err(err)?;
                let n = v.size();
                for ord in 0..n {
                    let src = MergeSourceValues::Byte(v.clone());
                    f.l(i64::from(src.ord_to_doc(ord).map_err(err)?));
                    f.b(v.vector(ord).map_err(err)?);
                }
                (MergeSourceValues::Byte(v), n)
            }
        };
        drop(values);
        let (s, n) = (seg.name(), &fi.name);
        let vec_line = format!("vec {s} {n} n={count} {}", f.hex());

        let mod3 = mod_filter(seg.max_doc(), 3);
        let mod89 = mod_filter(seg.max_doc(), 89);
        fn make<'d>(
            flat: &FlatVectorsReader<'d>,
            hnsw: &Option<GraphReader<'d>>,
            seg: &'d Segment,
            filter: Option<&'d FixedBitSet>,
        ) -> lucene_search::vector_query::VectorsInput<'d> {
            lucene_search::vector_query::VectorsInput {
                flat: flat.clone(),
                hnsw: hnsw.clone(),
                field_infos: &seg.field_infos,
                live_docs: seg.live.as_ref(),
                filter,
                max_doc: seg.max_doc(),
            }
        }
        let input = |filter| make(&flat, &hnsw, seg, filter);
        let leaf = |filter| {
            [KnnSegment {
                vectors: make(&flat, &hnsw, seg, filter),
                doc_base: 0,
            }]
        };
        let dim = fi.vector_dimension as usize;
        let by_doc = |mut hits: Vec<lucene_search::ScoreDoc>| {
            hits.sort_by_key(|h| h.doc_id);
            hits_str(&hits)
        };
        let thr =
            |knn: &[lucene_search::ScoreDoc]| knn.get(4).or(knn.last()).map_or(0.0f32, |h| h.score);
        let lines = match fi.vector_encoding {
            VectorEncoding::Float32 => {
                let q: Vec<f32> = (0..dim).map(|k| ((k + 1) as f64).sin() as f32).collect();
                let knn = |k: usize| KnnFloatVectorQuery::new(n.clone(), q.clone(), k).map_err(err);
                let hits =
                    vq::search_knn_float_vector_query(&input(None), &knn(10)?).map_err(err)?;
                let knnf =
                    vq::search_knn_float_vector_query_multi_segment(&leaf(Some(&mod3)), &knn(10)?)
                        .map_err(err)?;
                let knne =
                    vq::search_knn_float_vector_query_multi_segment(&leaf(Some(&mod89)), &knn(20)?)
                        .map_err(err)?;
                let patience = vq::search_patience_knn_float_vector_query_multi_segment(
                    &leaf(None),
                    &PatienceKnnVectorQuery::new(knn(10)?, 0.5, 2),
                )
                .map_err(err)?;
                let t = thr(&hits);
                let sim = FloatVectorSimilarityQuery::new(
                    n.clone(),
                    q.clone(),
                    t,
                    FloatVectorSimilarityQuery::DEFAULT_DECAY,
                )
                .map_err(err)?;
                let vsim = vq::float_vector_similarity_hits(&leaf(None), &sim).map_err(err)?;
                let vsimf =
                    vq::float_vector_similarity_hits(&leaf(Some(&mod3)), &sim).map_err(err)?;
                (hits, knnf, knne, patience, t, vsim, vsimf)
            }
            VectorEncoding::Byte => {
                let q: Vec<u8> = (0..dim)
                    .map(|k| ((k as i32) * 37 - 100) as i8 as u8)
                    .collect();
                let knn = |k: usize| KnnByteVectorQuery::new(n.clone(), q.clone(), k).map_err(err);
                let hits =
                    vq::search_knn_byte_vector_query(&input(None), &knn(10)?).map_err(err)?;
                let knnf =
                    vq::search_knn_byte_vector_query_multi_segment(&leaf(Some(&mod3)), &knn(10)?)
                        .map_err(err)?;
                let knne =
                    vq::search_knn_byte_vector_query_multi_segment(&leaf(Some(&mod89)), &knn(20)?)
                        .map_err(err)?;
                let patience = vq::search_patience_knn_byte_vector_query_multi_segment(
                    &leaf(None),
                    &PatienceKnnVectorQuery::new(knn(10)?, 0.5, 2),
                )
                .map_err(err)?;
                let t = thr(&hits);
                let sim = ByteVectorSimilarityQuery::new(
                    n.clone(),
                    q.clone(),
                    t,
                    ByteVectorSimilarityQuery::DEFAULT_DECAY,
                )
                .map_err(err)?;
                let vsim = vq::byte_vector_similarity_hits(&leaf(None), &sim).map_err(err)?;
                let vsimf =
                    vq::byte_vector_similarity_hits(&leaf(Some(&mod3)), &sim).map_err(err)?;
                (hits, knnf, knne, patience, t, vsim, vsimf)
            }
        };
        let (hits, knnf, knne, patience, t, vsim, vsimf) = lines;
        Ok(vec![
            vec_line,
            format!("knn {s} {n} {}", hits_str(&hits)),
            format!("knnf {s} {n} {}", hits_str(&knnf)),
            format!("knne {s} {n} {}", hits_str(&knne)),
            format!("patience {s} {n} {}", hits_str(&patience)),
            format!("vsim {s} {n} thr={:x} {}", t.to_bits(), by_doc(vsim)),
            format!("vsimf {s} {n} thr={:x} {}", t.to_bits(), by_doc(vsimf)),
        ])
    })();
    match result {
        Ok(lines) => {
            for (key, line) in keys.into_iter().zip(lines) {
                out.insert(key, Ok(line));
            }
        }
        Err(e) => {
            for key in keys {
                out.insert(key, Err(e.clone()));
            }
        }
    }
}

/// Every line this port can produce for one segment, keyed by the line's
/// `kind segment [field]` prefix. A segment that does not open at all
/// reports every one of its keys the caller asks for as missing.
fn segment_lines(dir: &dyn Directory, seg: &Segment, out: &mut Lines) {
    let name = seg.name().to_string();
    for fi in &seg.field_infos.fields {
        out.insert(
            format!("field {name} {}", fi.name),
            Ok(field_line(&name, fi)),
        );
    }
    out.insert(format!("live {name}"), Ok(live_line(seg)));

    let postings = open_postings_files(dir, seg).and_then(|files| {
        let fields = blocktree::open(
            &files.tim,
            &files.tip,
            &files.tmd,
            &seg.field_infos,
            &seg.commit.segment_id,
            &files.suffix,
            seg.max_doc(),
        )
        .map_err(err)?;
        Ok((files, fields))
    });
    let norms = (|| -> Result<(norms::Norms, Input), String> {
        let nvm_name = seg.file_with(".nvm").ok_or("no .nvm")?;
        let nvm = dir.open(nvm_name).map_err(err)?;
        let nvd = dir
            .open(seg.file_with(".nvd").ok_or("no .nvd")?)
            .map_err(err)?;
        let suffix = seg.suffix_of(nvm_name, ".nvm");
        let (_, meta) = norms::parse_meta(&nvm, &seg.commit.segment_id, &suffix).map_err(err)?;
        norms::check_data_header_footer(&nvd, &seg.commit.segment_id, &suffix).map_err(err)?;
        Ok((meta, nvd))
    })();
    let dv = (|| -> Result<(doc_values::DocValuesMeta, Input), String> {
        let dvm_name = seg.file_with(".dvm").ok_or("no .dvm")?;
        let dvm = dir.open(dvm_name).map_err(err)?;
        let dvd = dir
            .open(seg.file_with(".dvd").ok_or("no .dvd")?)
            .map_err(err)?;
        let suffix = seg.suffix_of(dvm_name, ".dvm");
        let (_, meta) =
            doc_values::parse_meta(&dvm, &seg.commit.segment_id, &suffix, &seg.field_infos)
                .map_err(err)?;
        doc_values::check_data_header_footer(&dvd, &seg.commit.segment_id, &suffix).map_err(err)?;
        Ok((meta, dvd))
    })();
    let point_files = (|| -> Result<(Input, Input, Input), String> {
        let open = |ext: &str| -> Result<Input, String> {
            dir.open(seg.file_with(ext).ok_or(format!("no {ext}"))?)
                .map_err(err)
        };
        Ok((open(".kdm")?, open(".kdi")?, open(".kdd")?))
    })();
    let points_reader = point_files
        .as_ref()
        .map_err(Clone::clone)
        .and_then(|(m, i, d)| points::open(m, i, d, &seg.commit.segment_id, "").map_err(err));

    for fi in &seg.field_infos.fields {
        let f = &fi.name;
        if fi.index_options != IndexOptions::None {
            let line = match &postings {
                Ok((files, fields)) => postings_line(seg, fi, fields, files),
                Err(e) => Err(e.clone()),
            };
            out.insert(format!("postings {name} {f}"), line);
        }
        if has_norms(fi) {
            let line = match &norms {
                Ok((meta, nvd)) => norms_line(seg, fi, meta, nvd),
                Err(e) => Err(e.clone()),
            };
            out.insert(format!("norms {name} {f}"), line);
        }
        if fi.doc_values_type != DocValuesType::None {
            let line = match &dv {
                Ok((meta, dvd)) => dv_line(seg, fi, meta, dvd),
                Err(e) => Err(e.clone()),
            };
            out.insert(format!("dv {name} {f}"), line);
        }
        if fi.point_dimension_count > 0 {
            let line = match &points_reader {
                Ok(r) => points_line(seg, fi, r),
                Err(e) => Err(e.clone()),
            };
            out.insert(format!("points {name} {f}"), line);
        }
        if fi.vector_dimension > 0 {
            vec_lines(dir, seg, fi, out);
        }
    }
    out.insert(format!("stored {name}"), stored_line(dir, seg));
    out.insert(format!("tv {name}"), tv_line(dir, seg));
}

/// The key of one `expected.txt` line: `kind segment [field]`.
fn key_of(line: &str) -> String {
    let parts: Vec<&str> = line.split(' ').collect();
    match parts[0] {
        "commit" => "commit".into(),
        "seg" | "live" | "stored" | "tv" => format!("{} {}", parts[0], parts[1]),
        _ => format!("{} {} {}", parts[0], parts[1], parts[2]),
    }
}

/// Every line of `version`'s fixture, as `(key, expected, actual)`.
fn run_version(version: &str) -> Vec<(String, String, Result<String, String>)> {
    run_dir(version, &fixture_dir(version))
}

/// Every line of the fixture in `path`, as `(key, expected, actual)`.
fn run_dir(version: &str, path: &std::path::Path) -> Vec<(String, String, Result<String, String>)> {
    let expected = std::fs::read_to_string(path.join("expected.txt"))
        .unwrap_or_else(|e| panic!("{version}: expected.txt: {e}"));
    let dir = FsDirectory::open(path);
    let mut actual = Lines::new();
    let mut seg_errors: BTreeMap<String, String> = BTreeMap::new();
    match segment_infos::read_latest(&dir) {
        Ok(infos) => {
            actual.insert(
                "commit".into(),
                Ok(format!(
                    "commit gen={} version={}.{}.{} created={} min={}",
                    infos.generation,
                    infos.lucene_version.major,
                    infos.lucene_version.minor,
                    infos.lucene_version.bugfix,
                    infos.index_created_version_major,
                    infos.min_segment_lucene_version.as_ref().map_or(
                        "null".to_string(),
                        |v| format!("{}.{}.{}", v.major, v.minor, v.bugfix)
                    ),
                )),
            );
            for commit in &infos.segments {
                match open_segment(&dir, commit) {
                    Ok(seg) => {
                        actual.insert(format!("seg {}", seg.name()), Ok(seg_line(&seg)));
                        segment_lines(&dir, &seg, &mut actual);
                    }
                    Err(e) => {
                        seg_errors.insert(commit.segment_name.clone(), e.clone());
                        actual.insert(format!("seg {}", commit.segment_name), Err(e));
                    }
                }
            }
        }
        Err(e) => {
            actual.insert("commit".into(), Err(e.to_string()));
        }
    }
    expected
        .lines()
        .map(|line| {
            let key = key_of(line);
            let got = actual.remove(&key).unwrap_or_else(|| {
                // A line whose segment did not open inherits that failure.
                let seg = key.split(' ').nth(1).unwrap_or("");
                Err(match seg_errors.get(seg) {
                    Some(e) => format!("segment did not open: {e}"),
                    None => "not produced".to_string(),
                })
            });
            (key, line.to_string(), got)
        })
        .collect()
}

fn listed_as_failing(version: &str, key: &str) -> bool {
    let parts: Vec<&str> = key.split(' ').collect();
    let kind = parts[0];
    let field = parts.get(2).copied().unwrap_or("");
    EXPECTED_FAILURES
        .iter()
        .any(|&(v, k, f)| v == version && k == kind && (f == "*" || f == field))
}

#[test]
fn bwc_fixtures_match_lucene() {
    let mut unexpected = Vec::new();
    let mut stale = Vec::new();
    let mut summary = Vec::new();
    for version in VERSIONS {
        let lines = run_version(version);
        let (mut pass, mut fail) = (0, 0);
        for (key, expected, got) in &lines {
            let ok = matches!(got, Ok(g) if g == expected);
            let listed = listed_as_failing(version, key);
            if ok {
                pass += 1;
                if listed {
                    stale.push(format!(
                        "{version}: {key} passes but is in EXPECTED_FAILURES"
                    ));
                }
            } else {
                fail += 1;
                if !listed {
                    let got = match got {
                        Ok(g) => format!("got      {g}"),
                        Err(e) => format!("error    {e}"),
                    };
                    unexpected.push(format!("{version}: {key}\n  expected {expected}\n  {got}"));
                }
            }
        }
        summary.push(format!("{version}: {pass} pass, {fail} fail"));
    }
    for version in QUANTIZED_VERSIONS {
        let lines = run_dir(version, &quantized_fixture_dir(version));
        let mut pass = 0;
        for (key, expected, got) in &lines {
            if matches!(got, Ok(g) if g == expected) {
                pass += 1;
            } else {
                let got = match got {
                    Ok(g) => format!("got      {g}"),
                    Err(e) => format!("error    {e}"),
                };
                unexpected.push(format!(
                    "quantized {version}: {key}\n  expected {expected}\n  {got}"
                ));
            }
        }
        summary.push(format!(
            "quantized {version}: {pass} of {} lines",
            lines.len()
        ));
    }
    eprintln!("{}", summary.join("\n"));
    assert!(
        unexpected.is_empty() && stale.is_empty(),
        "{} unexpected mismatches, {} stale EXPECTED_FAILURES entries:\n{}\n{}",
        unexpected.len(),
        stale.len(),
        unexpected.join("\n"),
        stale.join("\n")
    );
}

// ---------------------------------------------------------------------------
// The normal read path: every fixture opened with `DirectoryReader` and
// searched. `BwcWrite` writes the same documents under every version (the
// digests above are identical from 9.0.0 to 10.4.0), and 10.4.0 is the
// current `Lucene104` codec whose search path is verified against Lucene by
// the rest of this crate's fixtures -- so every older version must return
// 10.4.0's hits and scores, bit for bit, through the same query code.

fn query_set() -> Vec<(String, lucene_search::BooleanQuery)> {
    use lucene_search::query::{
        DisjunctionMaxQuery, PointsRangeQuery, PrefixQuery, RegexpQuery, TermInSetQuery,
        WildcardQuery,
    };
    use lucene_search::{BooleanQuery, Clause, PhraseQuery, TermQuery};
    let term = |f: &str, t: &str| Clause::Term(TermQuery::new(f, t.as_bytes().to_vec()));
    let one = |name: &str, c: Clause| {
        let mut b = BooleanQuery::new();
        b.must.push(c);
        (name.to_string(), b)
    };
    let mut out = Vec::new();
    for f in ["body", "title", "off", "pay", "freqs", "docs"] {
        out.push(one(&format!("term {f}:alpha"), term(f, "alpha")));
        out.push(one(&format!("term {f}:zeta"), term(f, "zeta")));
        out.push(one(
            &format!("prefix {f}:e"),
            Clause::Prefix(PrefixQuery::new(f, b"e".to_vec())),
        ));
    }
    for f in ["body", "off", "pay", "title"] {
        out.push(one(
            &format!("phrase {f}:alpha beta"),
            Clause::Phrase(PhraseQuery::new(f, vec!["alpha", "beta"])),
        ));
        out.push(one(
            &format!("phrase~3 {f}:gamma alpha"),
            Clause::Phrase(PhraseQuery::new(f, vec!["gamma", "alpha"]).with_slop(3)),
        ));
    }
    out.push(one("id:17", term("id", "17")));
    out.push(one(
        "wildcard body:*ta",
        Clause::Wildcard(WildcardQuery::new("body", b"*ta".to_vec())),
    ));
    out.push(one(
        "regexp body:[a-e].*a",
        Clause::Regexp(RegexpQuery::new("body", "[a-e].*a")),
    ));
    out.push(one(
        "terms id:{1,2,3000,3399}",
        Clause::TermInSet(TermInSetQuery::new(
            "id",
            vec![
                b"1".to_vec(),
                b"2".to_vec(),
                b"3000".to_vec(),
                b"3399".to_vec(),
            ],
        )),
    ));
    out.push(one(
        "range lpt:[-2^54, 2^54]",
        Clause::PointsRange(PointsRangeQuery::new("lpt", -(1 << 54), 1 << 54)),
    ));
    out.push(one(
        "dismax body:beta|title:beta",
        DisjunctionMaxQuery::new([term("body", "beta"), term("title", "beta")], 0.1).into(),
    ));
    let mut b = BooleanQuery::new();
    b.should.push(term("body", "alpha"));
    b.should.push(term("body", "omega"));
    b.should.push(term("title", "delta"));
    out.push(("or body:alpha body:omega title:delta".into(), b));
    let mut b = BooleanQuery::new();
    b.must.push(term("body", "beta"));
    b.must.push(term("off", "gamma"));
    b.must_not.push(term("docs", "delta"));
    b.filter.push(Clause::PointsRange(PointsRangeQuery::new(
        "lpt",
        i64::MIN,
        0,
    )));
    out.push(("and +body:beta +off:gamma -docs:delta #lpt<=0".into(), b));
    out
}

/// Every query of [`query_set`] against one fixture, top 20 exact and
/// pruned, as `name mode total hits(doc:scorebits)`.
fn search_version(version: &str) -> Result<Vec<String>, String> {
    search_dir(&fixture_dir(version))
}

/// [`query_set`] over the index in `path`, exact and pruned: one line per
/// query and mode, `name mode total hits`.
fn search_dir(path: &std::path::Path) -> Result<Vec<String>, String> {
    use lucene_search::directory_reader::DirectoryReader;
    use lucene_search::field_norms::FieldNorms;
    use lucene_search::multi_segment::search_boolean_query_multi_segment_maxscore_counting;
    use std::collections::HashMap;
    let dir = FsDirectory::open(path);
    let reader = DirectoryReader::open(&dir).map_err(err)?;
    let mut opened = reader.open_segments().map_err(err)?;
    opened.open_points().map_err(err)?;
    let segments = opened.as_open_segments();
    let mut owned: Vec<HashMap<String, FieldNorms<'_>>> =
        (0..segments.len()).map(|_| HashMap::new()).collect();
    for f in ["body", "title", "off", "pay", "freqs"] {
        for (i, n) in reader.field_norms(f).into_iter().enumerate() {
            if let Some(n) = n {
                owned[i].insert(f.to_string(), n);
            }
        }
    }
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let mut out = Vec::new();
    for (name, q) in query_set() {
        for (mode, limit) in [("exact", u64::MAX), ("pruned", 1000)] {
            let (hits, total) = search_boolean_query_multi_segment_maxscore_counting(
                &segments, &q, &norms, 20, limit,
            )
            .map_err(|e| format!("{name}: {e}"))?;
            let hits = hits
                .iter()
                .map(|h| format!("{}:{:x}", h.doc_id, h.score.to_bits()))
                .collect::<Vec<_>>()
                .join(",");
            let total = if mode == "exact" {
                total.value.to_string()
            } else {
                "-".to_string()
            };
            out.push(format!("{name} {mode} {total} {hits}"));
        }
    }
    Ok(out)
}

#[test]
fn every_version_searches_like_the_current_codec() {
    let reference = search_version("10.4.0").expect("10.4.0 searches");
    // The reference itself must be worth comparing against: every query
    // but a lookup of one (possibly deleted) id matches something.
    for line in reference.iter().filter(|l| !l.starts_with("id:")) {
        let hits = line.rsplit(' ').next().unwrap_or("");
        assert!(!hits.is_empty(), "10.4.0 query matched nothing: {line}");
    }
    let mut failures = Vec::new();
    for version in VERSIONS.iter().filter(|v| **v != "10.4.0") {
        match search_version(version) {
            Ok(got) => {
                for (g, r) in got.iter().zip(&reference) {
                    if g != r {
                        failures.push(format!("{version}:\n  got      {g}\n  10.4.0   {r}"));
                    }
                }
            }
            Err(e) => failures.push(format!("{version}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_version_passes_check_index() {
    let mut failures = Vec::new();
    for version in VERSIONS {
        let dir = FsDirectory::open(fixture_dir(version));
        let results = lucene_index::check_index::check_directory(&dir)
            .unwrap_or_else(|e| panic!("{version}: {e}"));
        for r in results {
            for c in r.failures() {
                failures.push(format!(
                    "{version} {}: {} {}",
                    r.segment_name, c.name, c.message
                ));
            }
            // A pass is only worth something if the vector families ran:
            // every segment has an `fvec` field whose vectors must have been
            // read, whichever format wrote them, and `_0` (3,000 documents)
            // always carries a graph -- 10.4's `_1` is under
            // `HNSW_GRAPH_THRESHOLD` and has none.
            if r.segment_name.starts_with('_') {
                let mut families = vec!["vectors.values_decode:fvec"];
                if r.segment_name == "_0" {
                    families.push("hnsw.neighbors_on_level:fvec");
                }
                for family in families {
                    if !r.checks.iter().any(|c| c.name == family && c.passed()) {
                        failures.push(format!(
                            "{version} {}: {family} did not run",
                            r.segment_name
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// M8 T8.4, this side of it: a buffered delete by term resolves against an
/// old segment's own term dictionary, and every fixture, force-merged by
/// this port's `IndexWriter`, becomes one `Lucene104` segment -- every postings field
/// `Lucene104`, every vector field `Lucene99HnswVectorsFormat` -- that this
/// port's `CheckIndex` passes, holding the original's live documents: every
/// query of [`query_set`] matches as many documents as it did before. (Scores
/// differ: a merge drops the deleted documents the statistics counted.)
/// Real Lucene's verdict on the same merges -- `CheckIndex` and a
/// per-document comparison of every kind of content -- is
/// `scripts/verify-bwc-merge.sh`.
#[test]
fn every_version_force_merges_into_lucene104() {
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    let mut failures = Vec::new();
    for version in VERSIONS {
        let tmp = lucene_util::test_support::TempDir::new(&format!("bwc-merge-{version}"));
        for entry in std::fs::read_dir(fixture_dir(version)).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".txt") {
                std::fs::copy(entry.path(), tmp.path().join(&name)).unwrap();
            }
        }
        let dir = FsDirectory::open(tmp.path());
        fn open_writer(dir: &FsDirectory) -> IndexWriter<'_> {
            IndexWriter::open(
                dir,
                Vec::new(),
                "Lucene104",
                LuceneVersion {
                    major: 10,
                    minor: 5,
                    bugfix: 0,
                },
            )
            .unwrap()
        }
        let deleted = |dir: &FsDirectory| -> i32 {
            segment_infos::read_latest(dir)
                .unwrap()
                .segments
                .iter()
                .map(|s| s.del_count)
                .sum()
        };
        // `id:5` lives in the old `_0`; deleting it rewrites that segment's
        // `.liv` through its `Lucene90`..`Lucene104` postings.
        let deleted_before = deleted(&dir);
        {
            let mut w = open_writer(&dir);
            w.delete_documents_by_term(&[lucene_index::buffered_updates::Term {
                field: "id".to_string(),
                bytes: b"5".to_vec(),
            }])
            .unwrap();
            w.commit().unwrap();
        }
        if deleted(&dir) != deleted_before + 1 {
            failures.push(format!(
                "{version}: delete id:5 on the old segment matched nothing"
            ));
        }
        let before = search_dir(tmp.path()).unwrap();
        {
            let mut w = open_writer(&dir);
            w.force_merge(1)
                .unwrap_or_else(|e| panic!("{version}: force_merge: {e}"));
            w.commit().unwrap();
        }
        let infos = segment_infos::read_latest(&dir).unwrap();
        let segs: Vec<(&str, &str)> = infos
            .segments
            .iter()
            .map(|s| (s.segment_name.as_str(), s.codec_name.as_str()))
            .collect();
        if segs.len() != 1 || segs[0].1 != "Lucene104" {
            failures.push(format!("{version}: merged into {segs:?}"));
            continue;
        }
        let seg = open_segment(&dir, &infos.segments[0]).unwrap();
        for fi in &seg.field_infos.fields {
            for (key, want) in [
                ("PerFieldPostingsFormat.format", "Lucene104"),
                (
                    "PerFieldKnnVectorsFormat.format",
                    "Lucene99HnswVectorsFormat",
                ),
            ] {
                if let Some((_, got)) = fi.attributes.iter().find(|(k, _)| k == key) {
                    if got != want {
                        failures.push(format!("{version}: {} {key}={got}", fi.name));
                    }
                }
            }
        }
        for r in lucene_index::check_index::check_directory(&dir).unwrap() {
            for c in r.failures() {
                failures.push(format!("{version}: CheckIndex {} {}", c.name, c.message));
            }
        }
        let totals = |lines: Vec<String>| -> Vec<String> {
            lines
                .into_iter()
                .filter(|l| l.contains(" exact "))
                .map(|l| {
                    l.rsplit_once(' ')
                        .map(|(head, _)| head.to_string())
                        .unwrap()
                })
                .collect()
        };
        let before = totals(before);
        let after = totals(search_dir(tmp.path()).unwrap_or_else(|e| panic!("{version}: {e}")));
        for (b, a) in before.iter().zip(&after) {
            if b != a {
                failures.push(format!("{version}: before {b}, after {a}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Per vector field, how many live documents of the index in `dir` carry a
/// vector: each field read through its own per-field group's raw vectors.
fn live_vector_counts(dir: &FsDirectory) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for commit in segment_infos::read_latest(dir).unwrap().segments {
        let seg = open_segment(dir, &commit).unwrap();
        for fi in seg
            .field_infos
            .fields
            .iter()
            .filter(|f| f.vector_dimension > 0)
        {
            let (format, suffix, files) = vector_files(dir, &seg, fi).unwrap();
            assert!(
                RetiredHnswFormat::for_name(&format).is_none(),
                "{format}: not a group this helper reads"
            );
            let flat = FlatVectorsReader::open(
                &files["vemf"],
                &files["vec"],
                &seg.commit.segment_id,
                &suffix,
            )
            .unwrap();
            let n = match fi.vector_encoding {
                VectorEncoding::Float32 => {
                    let v = flat.float_vector_values(fi.number).unwrap();
                    (0..v.size())
                        .filter(|&o| seg.is_live(v.ord_to_doc(o).unwrap()))
                        .count()
                }
                VectorEncoding::Byte => {
                    let v = flat.byte_vector_values(fi.number).unwrap();
                    (0..v.size())
                        .filter(|&o| seg.is_live(v.ord_to_doc(o).unwrap()))
                        .count()
                }
            };
            *counts.entry(fi.name.clone()).or_insert(0) += n;
        }
    }
    counts
}

/// The quantized fixtures through this port's `CheckIndex` and `IndexWriter`:
/// every group opens and checks clean (the quantized codes included), and a
/// force merge turns them into one `Lucene104` segment whose vector fields are
/// all `Lucene99HnswVectorsFormat` and hold every live vector. Real Lucene's
/// verdict on the same merges is `scripts/verify-bwc-merge.sh`.
#[test]
fn quantized_fixtures_check_clean_and_force_merge_into_lucene104() {
    use lucene_index::index_writer::IndexWriter;
    use lucene_index::segment_info::LuceneVersion;
    let mut failures = Vec::new();
    for version in QUANTIZED_VERSIONS {
        let src = FsDirectory::open(quantized_fixture_dir(version));
        for r in lucene_index::check_index::check_directory(&src).unwrap() {
            for c in r.failures() {
                failures.push(format!(
                    "{version} {}: {} {}",
                    r.segment_name, c.name, c.message
                ));
            }
            if !r.segment_name.starts_with('_') {
                continue;
            }
            let seg = open_segment(
                &src,
                &segment_infos::read_latest(&src)
                    .unwrap()
                    .segments
                    .into_iter()
                    .find(|s| s.segment_name == r.segment_name)
                    .unwrap(),
            )
            .unwrap();
            for fi in seg
                .field_infos
                .fields
                .iter()
                .filter(|f| f.vector_dimension > 0)
            {
                let mut families = vec![format!("vectors.values_decode:{}", fi.name)];
                if fi.vector_encoding == VectorEncoding::Float32 {
                    families.push(format!("vectors.quantized:{}", fi.name));
                }
                for family in families {
                    if !r.checks.iter().any(|c| c.name == family && c.passed()) {
                        failures.push(format!(
                            "{version} {}: {family} did not run",
                            r.segment_name
                        ));
                    }
                }
            }
            if r.segment_name == "_0"
                && !r
                    .checks
                    .iter()
                    .any(|c| c.name.starts_with("hnsw.neighbors_on_level:") && c.passed())
            {
                failures.push(format!("{version} _0: no graph was checked"));
            }
        }
        let before = live_vector_counts(&src);

        let tmp = lucene_util::test_support::TempDir::new(&format!("bwc-q-merge-{version}"));
        for entry in std::fs::read_dir(quantized_fixture_dir(version)).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".txt") {
                std::fs::copy(entry.path(), tmp.path().join(&name)).unwrap();
            }
        }
        let dir = FsDirectory::open(tmp.path());
        {
            let mut w = IndexWriter::open(
                &dir,
                Vec::new(),
                "Lucene104",
                LuceneVersion {
                    major: 10,
                    minor: 5,
                    bugfix: 0,
                },
            )
            .unwrap();
            w.force_merge(1)
                .unwrap_or_else(|e| panic!("{version}: force_merge: {e}"));
            w.commit().unwrap();
        }
        let infos = segment_infos::read_latest(&dir).unwrap();
        if infos.segments.len() != 1 || infos.segments[0].codec_name != "Lucene104" {
            failures.push(format!(
                "{version}: merged into {} segments",
                infos.segments.len()
            ));
            continue;
        }
        let seg = open_segment(&dir, &infos.segments[0]).unwrap();
        for fi in seg
            .field_infos
            .fields
            .iter()
            .filter(|f| f.vector_dimension > 0)
        {
            let format = fi
                .attributes
                .iter()
                .find(|(k, _)| k == "PerFieldKnnVectorsFormat.format")
                .map(|(_, v)| v.as_str());
            if format != Some("Lucene99HnswVectorsFormat") {
                failures.push(format!("{version}: {} merged as {format:?}", fi.name));
            }
        }
        for r in lucene_index::check_index::check_directory(&dir).unwrap() {
            for c in r.failures() {
                failures.push(format!(
                    "{version}: merged CheckIndex {} {}",
                    c.name, c.message
                ));
            }
        }
        let after = live_vector_counts(&dir);
        if before != after {
            failures.push(format!(
                "{version}: live vectors {before:?} before, {after:?} after"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
