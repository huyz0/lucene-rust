//! Differential test for `lucene_search::reader` against
//! `fixtures/data/reader_api/manifest.properties` (`fixtures/src/GenReaderApi.java`).
//!
//! Every leaf-shaped reader Java wrote out -- each `SegmentReader` of a
//! three-segment index, `SlowCompositeCodecReaderWrapper` over them, two
//! `SortingCodecReader`s, `ParallelCompositeReader`'s leaves, a
//! `ParallelLeafReader` with separate stored-fields readers and a
//! `SlowCodecReaderWrapper` -- is written out here by the same [`dump`] over
//! the Rust port of that view, and every line must be Java's: field infos,
//! live docs, terms statistics, postings with positions/offsets/payloads,
//! norms, the five doc-values types, points, float and byte vectors, stored
//! fields and term vectors. Composite-level lines cover `MultiDocValues`,
//! `MultiTerms` (`intersect`, `impacts`), `MultiReader`, and where a counting
//! `QueryTimeout` stops each of `ExitableDirectoryReader`'s enumerations.
//!
//! Regenerate with `scripts/gen-fixtures.sh --only GenReaderApi`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use lucene_codecs::field_infos::{DocValuesType, FieldInfos, IndexOptions, VectorEncoding};
use lucene_codecs::regexp::RegexpPattern;
use lucene_codecs::stored_fields::{Result as StoredResult, StoredFieldVisitor, VisitStatus};
use lucene_index::segment_info::{IndexSortField, IndexSortKind, StringMissingValue};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::multi_terms::MultiTerms;
use lucene_search::reader::exitable::{ExitableDirectoryReader, QueryTimeout};
use lucene_search::reader::multi_doc_values;
use lucene_search::reader::multi_reader::MultiReader;
use lucene_search::reader::parallel::{ParallelCompositeReader, ParallelLeafReader};
use lucene_search::reader::slow_codec::{SlowCodecReaderWrapper, SlowCompositeCodecReaderWrapper};
use lucene_search::reader::sorting::SortingCodecReader;
use lucene_search::reader::{
    merged_field_infos, BinaryDocValues, CodecReader, CompositeReader, IndexReader,
    IntersectVisitor, LeafReader, NumericDocValues, PostingsEnum, PostingsFlags, ReaderHandle,
    Relation, SortedDocValues, SortedNumericDocValues, SortedSetDocValues, Terms, NO_MORE_DOCS,
};
use lucene_store::FsDirectory;

fn root() -> String {
    format!(
        "{}/../../fixtures/data/reader_api",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn manifest() -> HashMap<String, String> {
    std::fs::read_to_string(format!("{}/manifest.properties", root()))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn open(name: &str) -> Arc<DirectoryReader> {
    let dir = FsDirectory::open(format!("{}/{name}", root()));
    Arc::new(DirectoryReader::open(&dir).unwrap())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn utf8(b: &[u8]) -> String {
    String::from_utf8(b.to_vec()).unwrap()
}

fn field_list(infos: &FieldInfos) -> String {
    let mut fields: Vec<_> = infos.fields.iter().collect();
    fields.sort_by_key(|f| f.number);
    fields
        .iter()
        .map(|f| format!("{}:{}", f.name, f.number))
        .collect::<Vec<_>>()
        .join(",")
}

fn postings(pe: &mut dyn PostingsEnum, positions: bool) -> String {
    let mut out = Vec::new();
    loop {
        let doc = pe.next_doc().unwrap();
        if doc == NO_MORE_DOCS {
            break;
        }
        let mut s = format!("{doc}/{}", pe.freq());
        if positions {
            let mut ps = Vec::new();
            for _ in 0..pe.freq() {
                let p = pe.next_position().unwrap();
                ps.push(format!(
                    "{p}@{}-{}#{}",
                    pe.start_offset(),
                    pe.end_offset(),
                    pe.payload().map(hex).unwrap_or_default()
                ));
            }
            s.push_str(&format!("[{}]", ps.join(" ")));
        }
        out.push(s);
    }
    out.join(" ")
}

fn terms_dump(t: &dyn Terms, with_postings: bool) -> String {
    let mut te = t.iterator().unwrap();
    terms_enum_dump(&mut *te, t.has_positions(), with_postings)
}

fn terms_enum_dump(
    te: &mut dyn lucene_search::reader::TermsEnum,
    pos: bool,
    with_postings: bool,
) -> String {
    let mut out = Vec::new();
    while let Some(term) = te.next().unwrap() {
        let term = utf8(term);
        let mut s = format!(
            "{term}:{}:{}",
            te.doc_freq().unwrap(),
            te.total_term_freq().unwrap()
        );
        if with_postings {
            let flags = if pos {
                PostingsFlags::All
            } else {
                PostingsFlags::Freqs
            };
            let mut pe = te.postings(flags).unwrap();
            s.push(':');
            s.push_str(&postings(&mut *pe, pos));
        }
        out.push(s);
    }
    out.join(";")
}

fn terms_stats(t: &dyn Terms) -> String {
    let b = |x: bool| if x { "1" } else { "0" };
    format!(
        "{}|{}|{}|{}|{}|{}|{}{}{}{}",
        t.size(),
        t.sum_total_term_freq(),
        t.sum_doc_freq(),
        t.doc_count(),
        t.min().unwrap().map(|m| utf8(&m)).unwrap_or_default(),
        t.max().unwrap().map(|m| utf8(&m)).unwrap_or_default(),
        b(t.has_freqs()),
        b(t.has_positions()),
        b(t.has_offsets()),
        b(t.has_payloads()),
    )
}

fn numeric(v: &mut dyn NumericDocValues) -> String {
    let mut out = Vec::new();
    loop {
        let d = v.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        out.push(format!("{d}:{}", v.long_value()));
    }
    out.join(",")
}

fn binary(v: &mut dyn BinaryDocValues) -> String {
    let mut out = Vec::new();
    loop {
        let d = v.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        out.push(format!("{d}:{}", hex(v.binary_value())));
    }
    out.join(",")
}

fn sorted(v: &mut dyn SortedDocValues) -> String {
    let mut out = Vec::new();
    loop {
        let d = v.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        out.push(format!("{d}:{}", v.ord_value()));
    }
    let terms: Vec<String> = (0..v.value_count())
        .map(|o| utf8(&v.lookup_ord(o).unwrap()))
        .collect();
    format!("{}|{}:{}", out.join(","), v.value_count(), terms.join("/"))
}

fn sorted_numeric(v: &mut dyn SortedNumericDocValues) -> String {
    let mut out = Vec::new();
    loop {
        let d = v.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        let vals: Vec<String> = (0..v.doc_value_count())
            .map(|_| v.next_value().unwrap().to_string())
            .collect();
        out.push(format!("{d}:[{}]", vals.join("|")));
    }
    out.join(",")
}

fn sorted_set(v: &mut dyn SortedSetDocValues) -> String {
    let mut out = Vec::new();
    loop {
        let d = v.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            break;
        }
        let vals: Vec<String> = (0..v.doc_value_count())
            .map(|_| v.next_ord().unwrap().to_string())
            .collect();
        out.push(format!("{d}:[{}]", vals.join("|")));
    }
    let terms: Vec<String> = (0..v.value_count())
        .map(|o| utf8(&v.lookup_ord(o).unwrap()))
        .collect();
    format!("{}|{}:{}", out.join(","), v.value_count(), terms.join("/"))
}

/// Every stored field of a document, as `name=type:value`.
struct StoredDump<'a> {
    infos: &'a FieldInfos,
    parts: Vec<String>,
}

impl StoredDump<'_> {
    fn name(&self, n: i32) -> String {
        self.infos
            .field_by_number(n)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("#{n}"))
    }
}

impl StoredFieldVisitor for StoredDump<'_> {
    fn needs_field(&mut self, _: i32) -> StoredResult<VisitStatus> {
        Ok(VisitStatus::Yes)
    }
    fn string_field(&mut self, n: i32, v: &str) -> StoredResult<()> {
        let name = self.name(n);
        self.parts.push(format!("{name}=s:{v}"));
        Ok(())
    }
    fn binary_field(&mut self, n: i32, v: &[u8]) -> StoredResult<()> {
        let name = self.name(n);
        self.parts.push(format!("{name}=b:{}", hex(v)));
        Ok(())
    }
    fn int_field(&mut self, n: i32, v: i32) -> StoredResult<()> {
        let name = self.name(n);
        self.parts.push(format!("{name}=i:{v}"));
        Ok(())
    }
    fn long_field(&mut self, n: i32, v: i64) -> StoredResult<()> {
        let name = self.name(n);
        self.parts.push(format!("{name}=l:{v}"));
        Ok(())
    }
    fn float_field(&mut self, n: i32, v: f32) -> StoredResult<()> {
        let name = self.name(n);
        self.parts.push(format!("{name}=f:{:x}", v.to_bits()));
        Ok(())
    }
    fn double_field(&mut self, n: i32, v: f64) -> StoredResult<()> {
        let name = self.name(n);
        self.parts.push(format!("{name}=d:{:x}", v.to_bits()));
        Ok(())
    }
}

fn stored<R: IndexReader + ?Sized>(r: &R, infos: &FieldInfos, doc: i32) -> String {
    let mut v = StoredDump {
        infos,
        parts: Vec::new(),
    };
    r.stored_document(doc, &mut v).unwrap();
    v.parts.join(";")
}

fn term_vectors(r: &dyn LeafReader, doc: i32) -> String {
    let Some(d) = r.term_vectors(doc).unwrap() else {
        return "null".into();
    };
    let infos = r.field_infos();
    let mut by_name = BTreeMap::new();
    for f in d.fields {
        let name = infos.field_by_number(f.field_number).unwrap().name.clone();
        let mut terms = f.terms.clone();
        terms.sort_by(|a, b| a.term.cmp(&b.term));
        let body: Vec<String> = terms
            .iter()
            .map(|t| {
                let occ: Vec<String> = (0..t.freq as usize)
                    .map(|k| {
                        let p = if f.has_positions {
                            t.positions.as_ref().map_or(-1, |p| p[k])
                        } else {
                            -1
                        };
                        let (s, e) = if f.has_offsets {
                            (
                                t.start_offsets.as_ref().map_or(-1, |o| o[k]),
                                t.end_offsets.as_ref().map_or(-1, |o| o[k]),
                            )
                        } else {
                            (-1, -1)
                        };
                        let pay = if f.has_payloads {
                            t.payloads.as_ref().map(|p| hex(&p[k])).unwrap_or_default()
                        } else {
                            String::new()
                        };
                        format!("{p}@{s}-{e}#{pay}")
                    })
                    .collect();
                format!("{}/{}[{}]", utf8(&t.term), t.freq, occ.join(" "))
            })
            .collect();
        by_name.insert(name.clone(), format!("{name}{{{}}}", body.join(" ")));
    }
    by_name.into_values().collect::<Vec<_>>().join(";")
}

/// Accepts every point, asking for its value.
struct AllPoints(Vec<String>);

impl IntersectVisitor for AllPoints {
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
    fn visit(&mut self, _: i32) {
        panic!("inside cells are not asked for");
    }
    fn visit_with_value(&mut self, doc: i32, v: &[u8]) {
        self.0.push(format!("{doc}:{}", hex(v)));
    }
}

fn dump(r: &dyn LeafReader, p: &str, out: &mut BTreeMap<String, String>) {
    let mut put = |k: String, v: String| {
        out.insert(k, v);
    };
    put(format!("{p}.max_doc"), r.max_doc().to_string());
    put(format!("{p}.num_docs"), r.num_docs().to_string());
    let deleted = match r.live_docs() {
        None => "none".to_string(),
        Some(live) => (0..r.max_doc())
            .filter(|&d| !live.get_doc(d))
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(","),
    };
    put(format!("{p}.deleted"), deleted);
    let infos = r.field_infos();
    put(format!("{p}.fields"), field_list(infos));
    put(format!("{p}.sorted"), r.index_sort().is_some().to_string());
    let mut by_name: Vec<_> = infos.fields.iter().collect();
    by_name.sort_by(|a, b| a.name.cmp(&b.name));
    for fi in by_name {
        let f = fi.name.as_str();
        if fi.index_options != IndexOptions::None {
            if let Some(t) = r.terms(f).unwrap() {
                put(format!("{p}.terms.{f}"), terms_stats(&*t));
                put(format!("{p}.postings.{f}"), terms_dump(&*t, true));
            }
            if !fi.omit_norms {
                if let Some(mut n) = r.norm_values(f).unwrap() {
                    put(format!("{p}.norms.{f}"), numeric(&mut *n));
                }
            }
        }
        match fi.doc_values_type {
            DocValuesType::Numeric => put(
                format!("{p}.dv.{f}"),
                numeric(&mut *r.numeric_doc_values(f).unwrap().unwrap()),
            ),
            DocValuesType::Binary => put(
                format!("{p}.dv.{f}"),
                binary(&mut *r.binary_doc_values(f).unwrap().unwrap()),
            ),
            DocValuesType::Sorted => put(
                format!("{p}.dv.{f}"),
                sorted(&mut *r.sorted_doc_values(f).unwrap().unwrap()),
            ),
            DocValuesType::SortedNumeric => put(
                format!("{p}.dv.{f}"),
                sorted_numeric(&mut *r.sorted_numeric_doc_values(f).unwrap().unwrap()),
            ),
            DocValuesType::SortedSet => put(
                format!("{p}.dv.{f}"),
                sorted_set(&mut *r.sorted_set_doc_values(f).unwrap().unwrap()),
            ),
            DocValuesType::None => {}
        }
        if fi.point_dimension_count > 0 {
            if let Some(pv) = r.point_values(f).unwrap() {
                let mut all = AllPoints(Vec::new());
                pv.intersect(&mut all).unwrap();
                all.0.sort();
                put(
                    format!("{p}.points.{f}"),
                    format!(
                        "{}|{}|{}|{}|{}|{}|{}|{}",
                        pv.num_dimensions(),
                        pv.num_index_dimensions(),
                        pv.bytes_per_dimension(),
                        pv.size(),
                        pv.doc_count(),
                        hex(pv.min_packed_value()),
                        hex(pv.max_packed_value()),
                        all.0.join(",")
                    ),
                );
            }
        }
        if fi.vector_dimension > 0 {
            let s: Vec<String> = if fi.vector_encoding == VectorEncoding::Float32 {
                let v = r.float_vector_values(f).unwrap().unwrap();
                (0..v.size())
                    .map(|o| {
                        let x = v.vector_value(o).unwrap();
                        let bits: Vec<String> =
                            x.iter().map(|f| format!("{:x}", f.to_bits())).collect();
                        format!("{}:{}", v.ord_to_doc(o).unwrap(), bits.join("/"))
                    })
                    .collect()
            } else {
                let v = r.byte_vector_values(f).unwrap().unwrap();
                (0..v.size())
                    .map(|o| {
                        format!(
                            "{}:{}",
                            v.ord_to_doc(o).unwrap(),
                            hex(&v.vector_value(o).unwrap())
                        )
                    })
                    .collect()
            };
            put(format!("{p}.vectors.{f}"), s.join(","));
        }
    }
    for doc in 0..r.max_doc() {
        put(format!("{p}.stored.{doc}"), stored(r, infos, doc));
        put(format!("{p}.tv.{doc}"), term_vectors(r, doc));
    }
}

/// Every manifest line under `prefix.` must be one the Rust dump wrote, and
/// agree with it; so must the reverse.
fn check(want: &HashMap<String, String>, got: &BTreeMap<String, String>, prefix: &str) {
    let dot = format!("{prefix}.");
    let mut n = 0;
    for (k, v) in want.iter().filter(|(k, _)| k.starts_with(&dot)) {
        assert_eq!(got.get(k), Some(v), "{k}");
        n += 1;
    }
    for k in got.keys().filter(|k| k.starts_with(&dot)) {
        assert!(want.contains_key(k), "{k} written by Rust only");
    }
    assert!(n > 0, "no lines for {prefix}");
}

#[test]
fn segment_readers_match_lucene() {
    let want = manifest();
    let r = open("multi");
    let mut got = BTreeMap::new();
    for (i, leaf) in r.leaves().iter().enumerate() {
        dump(leaf.reader, &format!("seg{i}"), &mut got);
        check(&want, &got, &format!("seg{i}"));
    }
}

#[test]
fn slow_composite_codec_reader_matches_lucene() {
    let want = manifest();
    let r = open("multi");
    let codecs: Vec<Arc<dyn CodecReader>> = r
        .segment_readers()
        .iter()
        .map(|s| Arc::new(s.clone()) as Arc<dyn CodecReader>)
        .collect();
    let slow = SlowCompositeCodecReaderWrapper::wrap(codecs).unwrap();
    let mut got = BTreeMap::new();
    dump(&*slow, "slow", &mut got);
    check(&want, &got, "slow");
}

#[test]
fn sorting_codec_readers_match_lucene() {
    let want = manifest();
    let r = open("multi");
    let segs = r.segment_readers();
    let mut got = BTreeMap::new();
    let by_rank = SortingCodecReader::wrap_sorted(
        Arc::new(segs[0].clone()),
        vec![IndexSortField::long("rank", false, None)],
    )
    .unwrap();
    dump(&by_rank, "sorted", &mut got);
    check(&want, &got, "sorted");
    let by_kw = SortingCodecReader::wrap_sorted(
        Arc::new(segs[1].clone()),
        vec![IndexSortField {
            field: "kw".into(),
            reverse: true,
            kind: IndexSortKind::String(StringMissingValue::None),
        }],
    )
    .unwrap();
    dump(&by_kw, "sorted_kw", &mut got);
    check(&want, &got, "sorted_kw");
}

#[test]
fn parallel_readers_match_lucene() {
    let want = manifest();
    let a = open("par_a");
    let b = open("par_b");
    let pc = ParallelCompositeReader::new(vec![
        a.clone() as Arc<dyn CompositeReader>,
        b.clone() as Arc<dyn CompositeReader>,
    ])
    .unwrap();
    let mut got = BTreeMap::new();
    for (i, leaf) in pc.leaves().iter().enumerate() {
        dump(leaf.reader, &format!("par{i}"), &mut got);
        check(&want, &got, &format!("par{i}"));
    }
    assert_eq!(pc.max_doc().to_string(), want["par.max_doc"]);
    assert_eq!(pc.num_docs().to_string(), want["par.num_docs"]);

    let a0: Arc<dyn LeafReader> = Arc::new(a.segment_readers()[0].clone());
    let b0: Arc<dyn LeafReader> = Arc::new(b.segment_readers()[0].clone());
    let stored = ParallelLeafReader::with_stored_fields_readers(
        vec![a0.clone(), b0.clone()],
        vec![b0.clone()],
    )
    .unwrap();
    dump(&stored, "parstored", &mut got);
    check(&want, &got, "parstored");

    let slow = SlowCodecReaderWrapper::wrap(Arc::new(
        ParallelLeafReader::new(vec![b0.clone(), a0.clone()]).unwrap(),
    ));
    dump(&*slow, "slowpar", &mut got);
    check(&want, &got, "slowpar");
}

#[test]
fn multi_reader_matches_lucene() {
    let want = manifest();
    let m = open("multi");
    let a = open("par_a");
    let mr = MultiReader::new(vec![
        ReaderHandle::Composite(m as Arc<dyn CompositeReader>),
        ReaderHandle::Composite(a as Arc<dyn CompositeReader>),
    ])
    .unwrap();
    let leaves: Vec<String> = mr
        .leaves()
        .iter()
        .map(|l| format!("{}:{}", l.doc_base, l.reader.max_doc()))
        .collect();
    assert_eq!(leaves.join(","), want["mr.leaves"]);
    assert_eq!(mr.max_doc().to_string(), want["mr.max_doc"]);
    assert_eq!(mr.num_docs().to_string(), want["mr.num_docs"]);
    assert_eq!(
        mr.doc_freq("body", b"fox").unwrap().to_string(),
        want["mr.doc_freq.body.fox"]
    );
    assert_eq!(
        mr.doc_freq("id", b"p2").unwrap().to_string(),
        want["mr.doc_freq.id.p2"]
    );
    assert_eq!(
        mr.sum_doc_freq("id").unwrap().to_string(),
        want["mr.sum_doc_freq.id"]
    );
    assert_eq!(
        mr.doc_count("id").unwrap().to_string(),
        want["mr.doc_count.id"]
    );
    assert_eq!(
        mr.sum_total_term_freq("body").unwrap().to_string(),
        want["mr.sum_ttf.body"]
    );
    assert_eq!(
        mr.total_term_freq("body", b"fox").unwrap().to_string(),
        want["mr.ttf.body.fox"]
    );
    let merged = merged_field_infos(&mr.leaves());
    assert_eq!(field_list(&merged), want["mr.fields"]);
    for doc in [0, 14, 15, 19] {
        assert_eq!(
            stored(&mr, &merged, doc),
            want[&format!("mr.stored.{doc}")],
            "{doc}"
        );
    }
}

#[test]
fn multi_doc_values_and_terms_match_lucene() {
    let want = manifest();
    let r = open("multi");
    let r = &*r;
    let key = |k: &str| want[&format!("multi.{k}")].clone();
    assert_eq!(field_list(&merged_field_infos(&r.leaves())), key("fields"));
    assert_eq!(
        numeric(
            &mut *multi_doc_values::numeric_values(r, "rank")
                .unwrap()
                .unwrap()
        ),
        key("mdv.rank")
    );
    assert_eq!(
        binary(&mut *multi_doc_values::binary_values(r, "bin").unwrap().unwrap()),
        key("mdv.bin")
    );
    assert_eq!(
        sorted(&mut *multi_doc_values::sorted_values(r, "kw").unwrap().unwrap()),
        key("mdv.kw")
    );
    assert_eq!(
        sorted_numeric(
            &mut *multi_doc_values::sorted_numeric_values(r, "nums")
                .unwrap()
                .unwrap()
        ),
        key("mdv.nums")
    );
    assert_eq!(
        sorted_set(
            &mut *multi_doc_values::sorted_set_values(r, "tags")
                .unwrap()
                .unwrap()
        ),
        key("mdv.tags")
    );
    assert_eq!(
        numeric(&mut *multi_doc_values::norm_values(r, "body").unwrap().unwrap()),
        key("mnorms.body")
    );
    for f in ["body", "pay", "kw"] {
        let t = MultiTerms::get_terms(r, f).unwrap().unwrap();
        assert_eq!(terms_stats(&*t), key(&format!("mterms.{f}")), "{f}");
        assert_eq!(terms_dump(&*t, true), key(&format!("mpostings.{f}")), "{f}");
    }
    let body = MultiTerms::get_terms(r, "body").unwrap().unwrap();
    let dfa = RegexpPattern::new(b"f.*|qu[a-z]*|b.*")
        .unwrap()
        .to_dfa()
        .unwrap();
    let mut te = body.intersect(&dfa, None).unwrap();
    assert_eq!(
        terms_enum_dump(&mut *te, false, false),
        key("intersect.body")
    );
    let mut te = body.intersect(&dfa, Some(b"fox")).unwrap();
    assert_eq!(
        terms_enum_dump(&mut *te, false, false),
        key("intersect_from_fox.body")
    );

    let mut te = body.iterator().unwrap();
    assert!(te.try_seek_exact(b"fox").unwrap());
    let mut ie = te.impacts(PostingsFlags::Freqs).unwrap();
    let upto = ie.advance_shallow(0).unwrap();
    let levels = ie.impacts();
    let got = format!(
        "{}|{}|{}|{}|{}|{}",
        levels.len(),
        levels[0].0,
        levels[0].1.len(),
        levels[0].1[0].freq,
        levels[0].1[0].norm,
        postings(&mut *ie, false)
    );
    assert_eq!(upto, levels[0].0);
    assert_eq!(got, key("impacts.body.fox"));
}

/// Exits on the `exit_at`-th `should_exit` call.
#[derive(Debug)]
struct Countdown {
    exit_at: u32,
    calls: std::sync::atomic::AtomicU32,
}

impl QueryTimeout for Countdown {
    fn should_exit(&self) -> bool {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 >= self.exit_at
    }
}

fn exitable(
    base: &Arc<DirectoryReader>,
    exit_at: u32,
) -> lucene_search::reader::filter::FilterDirectoryReader {
    ExitableDirectoryReader::wrap(
        base.clone(),
        Arc::new(Countdown {
            exit_at,
            calls: Default::default(),
        }),
    )
    .unwrap()
}

fn outcome(r: lucene_search::Result<()>, steps: i32, reached: i32) -> String {
    match r {
        Ok(()) => format!("end:{steps}:{reached}"),
        Err(lucene_search::Error::ExitingReader(_)) => format!("exit:{steps}:{reached}"),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn exitable_directory_reader_stops_where_lucene_does() {
    let want = manifest();
    let base = open("exitable");
    for exit_at in 1..=4u32 {
        let p = |k: &str| want[&format!("exit.{exit_at}.{k}")].clone();

        let r = exitable(&base, exit_at);
        let terms = r.leaves()[0].reader.terms("t").unwrap().unwrap();
        let mut n = 0;
        let res = (|| -> lucene_search::Result<()> {
            let mut te = terms.iterator()?;
            while te.next()?.is_some() {
                n += 1;
            }
            Ok(())
        })();
        let got = match res {
            Ok(()) => format!("end:{n}"),
            Err(lucene_search::Error::ExitingReader(_)) => format!("exit:{n}"),
            Err(e) => panic!("{e}"),
        };
        assert_eq!(got, p("terms"), "terms, exit at {exit_at}");

        let r = exitable(&base, exit_at);
        let leaf = r.leaves()[0].reader;
        let (mut steps, mut reached) = (0, -1);
        let mut v = leaf.numeric_doc_values("n").unwrap().unwrap();
        let res = (|| loop {
            let d = v.next_doc()?;
            if d == NO_MORE_DOCS {
                return Ok(());
            }
            reached = d;
            steps += 1;
        })();
        assert_eq!(
            outcome(res, steps, reached),
            p("numeric_next"),
            "numeric {exit_at}"
        );

        let r = exitable(&base, exit_at);
        let leaf = r.leaves()[0].reader;
        let (mut steps, mut reached) = (0, -1);
        let mut v = leaf.sorted_doc_values("s").unwrap().unwrap();
        let res = (|| {
            let mut doc = 0;
            while doc < leaf.max_doc() {
                v.advance_exact(doc)?;
                reached = doc;
                steps += 1;
                doc += 3;
            }
            Ok(())
        })();
        assert_eq!(
            outcome(res, steps, reached),
            p("sorted_exact"),
            "sorted {exit_at}"
        );

        let r = exitable(&base, exit_at);
        let leaf = r.leaves()[0].reader;
        let (mut steps, mut reached) = (0, -1);
        let mut v = leaf.binary_doc_values("b").unwrap().unwrap();
        let res = (|| {
            let mut target = 0;
            while target < leaf.max_doc() {
                reached = v.advance(target)?;
                steps += 1;
                target += 450;
            }
            Ok(())
        })();
        assert_eq!(
            outcome(res, steps, reached),
            p("binary_advance"),
            "binary {exit_at}"
        );

        let r = exitable(&base, exit_at);
        let leaf = r.leaves()[0].reader;
        let (mut steps, mut reached) = (0, -1);
        let mut v = leaf.sorted_numeric_doc_values("sn").unwrap().unwrap();
        let res = (|| {
            let mut target = 0;
            while target < leaf.max_doc() {
                reached = v.advance(target)?;
                steps += 1;
                target += 450;
            }
            Ok(())
        })();
        assert_eq!(
            outcome(res, steps, reached),
            p("sorted_numeric_advance"),
            "sorted numeric {exit_at}"
        );

        let r = exitable(&base, exit_at);
        let leaf = r.leaves()[0].reader;
        let (mut steps, mut reached) = (0, -1);
        let mut v = leaf.sorted_set_doc_values("ss").unwrap().unwrap();
        let res = (|| loop {
            let d = v.next_doc()?;
            if d == NO_MORE_DOCS {
                return Ok(());
            }
            reached = d;
            steps += 1;
        })();
        assert_eq!(
            outcome(res, steps, reached),
            p("sorted_set_next"),
            "sorted set {exit_at}"
        );
    }
}
