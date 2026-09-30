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
/// Vectors (`vec`/`knn`) before 9.11 need the retired `Lucene90`..`Lucene95`
/// HNSW readers, which M8 ports separately from everything else here.
const EXPECTED_FAILURES: &[(&str, &str, &str)] = &[
    ("9.0.0", "vec", "*"),
    ("9.0.0", "knn", "*"),
    ("9.1.0", "vec", "*"),
    ("9.1.0", "knn", "*"),
    ("9.3.0", "vec", "*"),
    ("9.3.0", "knn", "*"),
    ("9.4.2", "vec", "*"),
    ("9.4.2", "knn", "*"),
    ("9.8.0", "vec", "*"),
    ("9.8.0", "knn", "*"),
];

fn fixture_dir(version: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/bwc")
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
    let tvd_name = seg.file_with(".tvd").ok_or("no .tvd")?;
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

fn vec_lines(dir: &dyn Directory, seg: &Segment, fi: &FieldInfo, out: &mut Lines) {
    let vec_key = format!("vec {} {}", seg.name(), fi.name);
    let knn_key = format!("knn {} {}", seg.name(), fi.name);
    let result = (|| -> Result<(String, String), String> {
        let vec_name = seg.file_with(".vec").ok_or("no .vec")?;
        let vemf = dir
            .open(seg.file_with(".vemf").ok_or("no .vemf")?)
            .map_err(err)?;
        let vec = dir.open(vec_name).map_err(err)?;
        let vem = dir
            .open(seg.file_with(".vem").ok_or("no .vem")?)
            .map_err(err)?;
        let vex = dir
            .open(seg.file_with(".vex").ok_or("no .vex")?)
            .map_err(err)?;
        let suffix = seg.suffix_of(vec_name, ".vec");
        let id = &seg.commit.segment_id;
        let flat = FlatVectorsReader::open(&vemf, &vec, id, &suffix).map_err(err)?;
        let hnsw = lucene_codecs::hnsw_vectors::HnswVectorsReader::open(&vem, &vex, id, &suffix)
            .map_err(err)?;
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
        let vec_line = format!("vec {} {} n={count} {}", seg.name(), fi.name, f.hex());

        let input = lucene_search::vector_query::VectorsInput {
            flat: flat.clone(),
            hnsw: Some(hnsw),
            field_infos: &seg.field_infos,
            live_docs: seg.live.as_ref(),
            filter: None,
            max_doc: seg.max_doc(),
        };
        let dim = fi.vector_dimension as usize;
        let hits = match fi.vector_encoding {
            VectorEncoding::Float32 => {
                let q: Vec<f32> = (0..dim).map(|k| ((k + 1) as f64).sin() as f32).collect();
                let query =
                    lucene_search::vector_query::KnnFloatVectorQuery::new(fi.name.clone(), q, 10)
                        .map_err(err)?;
                lucene_search::vector_query::search_knn_float_vector_query(&input, &query)
                    .map_err(err)?
            }
            VectorEncoding::Byte => {
                let q: Vec<u8> = (0..dim)
                    .map(|k| ((k as i32) * 37 - 100) as i8 as u8)
                    .collect();
                let query =
                    lucene_search::vector_query::KnnByteVectorQuery::new(fi.name.clone(), q, 10)
                        .map_err(err)?;
                lucene_search::vector_query::search_knn_byte_vector_query(&input, &query)
                    .map_err(err)?
            }
        };
        let hits = hits
            .iter()
            .map(|h| format!("{}:{:x}", h.doc_id, h.score.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        Ok((vec_line, format!("knn {} {} {hits}", seg.name(), fi.name)))
    })();
    match result {
        Ok((v, k)) => {
            out.insert(vec_key, Ok(v));
            out.insert(knn_key, Ok(k));
        }
        Err(e) => {
            out.insert(vec_key, Err(e.clone()));
            out.insert(knn_key, Err(e));
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
    let path = fixture_dir(version);
    let expected = std::fs::read_to_string(path.join("expected.txt"))
        .unwrap_or_else(|e| panic!("{version}: expected.txt: {e}"));
    let dir = FsDirectory::open(&path);
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
