use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{self, FieldInfo, FieldInfos, IndexOptions};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_index::indexing_chain::MAX_POSITION;
use lucene_search::reader::{self as read, Terms, TermsEnum, NO_MORE_DOCS};
use lucene_test_framework::codecs::asserting_postings::*;
use lucene_test_framework::codecs::cranky_postings as byte;

type Events = Rc<RefCell<Vec<String>>>;
fn event(events: &Events, text: impl Into<String>) {
    events.borrow_mut().push(text.into());
}
fn error(text: &str) -> Error {
    Error::Message(text.into())
}
fn outcome(action: impl FnOnce() -> Result<()>) -> String {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)) {
        Err(_) => "assert".into(),
        Ok(Err(error)) => format!("io:{error}"),
        Ok(Ok(())) => "ok".into(),
    }
}
#[derive(Clone)]
struct Case {
    options: IndexOptions,
    fields: Vec<String>,
    terms: Vec<Vec<u8>>,
    docs: Vec<i32>,
    positions: Vec<i32>,
    starts: Vec<i32>,
    ends: Vec<i32>,
    freq: i32,
    term_freqs: bool,
    term_positions: bool,
    term_offsets: bool,
    payloads: bool,
    null_terms: bool,
    delegate_fail: bool,
    delegate_scan: bool,
    iterator_fail: bool,
    next_fail: bool,
    postings_fail: bool,
    doc_fail: bool,
}
impl Default for Case {
    fn default() -> Self {
        Self {
            options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            fields: vec!["f".into()],
            terms: vec![vec![97], vec![128]],
            docs: vec![0, 2],
            positions: vec![0, 2],
            starts: vec![0, 4],
            ends: vec![2, 6],
            freq: 2,
            term_freqs: true,
            term_positions: true,
            term_offsets: true,
            payloads: true,
            null_terms: false,
            delegate_fail: false,
            delegate_scan: false,
            iterator_fail: false,
            next_fail: false,
            postings_fail: false,
            doc_fail: false,
        }
    }
}
impl Case {
    fn named(name: &str) -> Self {
        let mut c = Self::default();
        match name {
            "full" => {}
            "docs" => {
                c.options = IndexOptions::Docs;
                c.term_positions = false;
                c.term_offsets = false;
                c.payloads = false
            }
            "freqs" => {
                c.options = IndexOptions::DocsAndFreqs;
                c.term_positions = false;
                c.term_offsets = false;
                c.payloads = false
            }
            "positions" => {
                c.options = IndexOptions::DocsAndFreqsAndPositions;
                c.term_offsets = false;
                c.payloads = false
            }
            "positions-payloads" => {
                c.options = IndexOptions::DocsAndFreqsAndPositions;
                c.term_offsets = false
            }
            "offsets-no-payloads" => c.payloads = false,
            "empty-fields" => c.fields.clear(),
            "null-terms" => c.null_terms = true,
            "empty-terms" => c.terms.clear(),
            "empty-docs" => c.docs.clear(),
            "unknown-field" => c.fields = vec!["missing".into()],
            "duplicate-fields" => c.fields = vec!["f".into(), "f".into()],
            "reversed-fields" => c.fields = vec!["g".into(), "f".into()],
            "unicode-fields" => c.fields = vec!["\u{10000}".into(), "\u{e000}".into()],
            "unicode-fields-reversed" => c.fields = vec!["\u{e000}".into(), "\u{10000}".into()],
            "duplicate-terms" => c.terms = vec![vec![97], vec![97]],
            "reversed-terms" => c.terms = vec![vec![128], vec![97]],
            "empty-term-first" => c.terms = vec![vec![], vec![97]],
            "positions-mismatch" => c.term_positions = false,
            "offsets-mismatch" => c.term_offsets = false,
            "freqs-metadata-unchecked" => c.term_freqs = false,
            "doc-negative" => c.docs = vec![-1],
            "doc-duplicate" => c.docs = vec![0, 0],
            "doc-descending" => c.docs = vec![2, 0],
            "doc-maxdoc-unchecked" => c.docs = vec![100],
            "freq-zero" => c.freq = 0,
            "position-descending" => c.positions = vec![2, 1],
            "position-negative-one-allowed" => c.positions = vec![-1, 0],
            "position-too-negative" => c.positions = vec![-2, 0],
            "position-max" => c.positions = vec![0, MAX_POSITION],
            "position-over-max" => c.positions = vec![0, MAX_POSITION + 1],
            "offset-end-before-start" => c.ends = vec![-1, 6],
            "offset-descending" => c.starts = vec![2, 1],
            "offset-negative-one-allowed" => {
                c.starts = vec![-1, 0];
                c.ends = vec![-1, 2]
            }
            "delegate-failure-before-invalid-fields" => {
                c.delegate_fail = true;
                c.fields = vec!["missing".into()]
            }
            "delegate-enumerates-before-validation" => c.delegate_scan = true,
            "iterator-error" => c.iterator_fail = true,
            "next-error" => c.next_fail = true,
            "postings-error" => c.postings_fail = true,
            "doc-error" => c.doc_fail = true,
            _ => panic!("unknown case {name}"),
        }
        c
    }
}
struct ProbeFields {
    c: Rc<Case>,
    events: Events,
}
impl WriteFields for ProbeFields {
    fn field_names(&self) -> Result<Vec<String>> {
        event(&self.events, "fields");
        Ok(self.c.fields.clone())
    }
    fn terms(&self, name: &str) -> Result<Option<Box<dyn WriteTerms + '_>>> {
        event(&self.events, format!("terms:{name}"));
        Ok((!self.c.null_terms).then(|| {
            Box::new(ProbeTerms {
                c: self.c.clone(),
                events: self.events.clone(),
            }) as Box<dyn WriteTerms>
        }))
    }
}
struct ProbeTerms {
    c: Rc<Case>,
    events: Events,
}
impl WriteTerms for ProbeTerms {
    fn iterator(&self) -> Result<Box<dyn WriteTermsEnum + '_>> {
        event(&self.events, "iterator");
        if self.c.iterator_fail {
            return Err(error("iterator failure"));
        }
        Ok(Box::new(ProbeEnum {
            c: self.c.clone(),
            events: self.events.clone(),
            next: 0,
            previous: None,
        }))
    }
    fn has_positions(&self) -> bool {
        event(&self.events, "has-positions");
        self.c.term_positions
    }
    fn has_offsets(&self) -> bool {
        event(&self.events, "has-offsets");
        self.c.term_offsets
    }
    fn has_payloads(&self) -> bool {
        event(&self.events, "has-payloads");
        self.c.payloads
    }
}
struct ProbeEnum {
    c: Rc<Case>,
    events: Events,
    next: usize,
    previous: Option<usize>,
}
impl WriteTermsEnum for ProbeEnum {
    fn next(&mut self) -> Result<Option<Vec<u8>>> {
        event(&self.events, "term-next");
        if self.c.next_fail {
            return Err(error("next failure"));
        }
        let term = self.c.terms.get(self.next).cloned();
        self.next += 1;
        Ok(term)
    }
    fn postings(
        &mut self,
        reuse: Option<Box<dyn WritePostings>>,
        flags: u32,
    ) -> Result<Box<dyn WritePostings>> {
        let id = reuse
            .as_ref()
            .map(|value| &**value as *const dyn WritePostings as *const () as usize);
        event(
            &self.events,
            format!(
                "postings:{flags}:{}:{}",
                reuse.is_some(),
                id == self.previous
            ),
        );
        if self.c.postings_fail {
            return Err(error("postings failure"));
        }
        let value = Box::new(ProbePostings {
            c: self.c.clone(),
            events: self.events.clone(),
            next: 0,
            position: 0,
        });
        self.previous = Some(&*value as *const ProbePostings as usize);
        Ok(value)
    }
}
struct ProbePostings {
    c: Rc<Case>,
    events: Events,
    next: usize,
    position: usize,
}
impl WritePostings for ProbePostings {
    fn next_doc(&mut self) -> Result<i32> {
        event(&self.events, "doc-next");
        if self.c.doc_fail {
            return Err(error("doc failure"));
        }
        self.position = 0;
        let doc = self.c.docs.get(self.next).copied().unwrap_or(NO_MORE_DOCS);
        self.next += 1;
        Ok(doc)
    }
    fn freq(&mut self) -> Result<i32> {
        event(&self.events, "freq");
        Ok(self.c.freq)
    }
    fn next_position(&mut self) -> Result<i32> {
        event(&self.events, "position");
        let p = self.c.positions[self.position];
        self.position += 1;
        Ok(p)
    }
    fn start_offset(&mut self) -> Result<i32> {
        event(&self.events, "start");
        Ok(self.c.starts[self.position - 1])
    }
    fn end_offset(&mut self) -> Result<i32> {
        event(&self.events, "end");
        Ok(self.c.ends[self.position - 1])
    }
    fn payload(&mut self) -> Result<Option<Vec<u8>>> {
        event(&self.events, "payload");
        Ok(None)
    }
}
struct ProbeConsumer {
    c: Rc<Case>,
    events: Events,
    identity: usize,
    closes: usize,
    fail_close: usize,
}
impl FieldsConsumer for ProbeConsumer {
    fn write(&mut self, fields: &dyn WriteFields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        let same = fields as *const dyn WriteFields as *const () as usize == self.identity;
        event(&self.events, format!("delegate:{same}:{}", norms.is_none()));
        if self.c.delegate_fail {
            return Err(error("delegate failure"));
        }
        if self.c.delegate_scan {
            for name in fields.field_names()? {
                if let Some(terms) = fields.terms(&name)? {
                    let mut e = terms.iterator()?;
                    while e.next()?.is_some() {}
                }
            }
        }
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.closes += 1;
        event(&self.events, format!("close:{}", self.closes));
        if self.closes == self.fail_close {
            Err(error("close failure"))
        } else {
            Ok(())
        }
    }
}
#[test]
fn actual_java_consumer_assertion_order_flags_reuse_errors_and_double_close() {
    let text = include_str!("../../../fixtures/testframework/data/asserting_postings_consumer.tsv");
    let mut count = 0;
    for line in text.lines() {
        let row: Vec<_> = line.split('\t').collect();
        let name = row[0];
        let events = Events::default();
        let c = Rc::new(if name.starts_with("close-") {
            Case::default()
        } else {
            Case::named(name)
        });
        let fields = Box::new(ProbeFields {
            c: c.clone(),
            events: events.clone(),
        });
        let infos = FieldInfos::new(
            ["f", "g", "\u{10000}", "\u{e000}"]
                .into_iter()
                .enumerate()
                .map(|(i, name)| FieldInfo::new(name, i as i32).with_index_options(c.options))
                .collect(),
        )
        .unwrap();
        let delegate = ProbeConsumer {
            c,
            events: events.clone(),
            identity: &*fields as *const ProbeFields as usize,
            closes: 0,
            fail_close: name
                .strip_prefix("close-")
                .and_then(|n| n.parse().ok())
                .unwrap_or(0),
        };
        let mut wrapper = AssertingFieldsConsumer::new(delegate, &infos);
        let status = outcome(|| {
            if name.starts_with("close-") {
                wrapper.close()
            } else {
                wrapper.write(&*fields, None)
            }
        });
        assert_eq!(status, row[1], "{name}");
        assert_eq!(events.borrow().join(","), row[2], "{name}");
        count += 1;
    }
    assert_eq!(count, 43);
}

struct ProbeReadTerms {
    c: Rc<Case>,
    events: Events,
}
impl Terms for ProbeReadTerms {
    fn iterator(&self) -> lucene_search::Result<Box<dyn TermsEnum + '_>> {
        event(&self.events, "iterator");
        Ok(Box::new(ProbeReadEnum {
            c: self.c.clone(),
            events: self.events.clone(),
            next: 0,
            current: None,
        }))
    }
    fn size(&self) -> i64 {
        event(&self.events, "term-size");
        self.c.terms.len() as i64
    }
    fn sum_total_term_freq(&self) -> i64 {
        event(&self.events, "sum-total");
        (self.c.terms.len() * self.c.docs.len()) as i64 * i64::from(self.c.freq)
    }
    fn sum_doc_freq(&self) -> i64 {
        event(&self.events, "sum-doc");
        (self.c.terms.len() * self.c.docs.len()) as i64
    }
    fn doc_count(&self) -> i32 {
        event(&self.events, "doc-count");
        self.c.docs.len() as i32
    }
    fn has_freqs(&self) -> bool {
        event(&self.events, "has-freqs");
        self.c.term_freqs
    }
    fn has_positions(&self) -> bool {
        event(&self.events, "has-positions");
        self.c.term_positions
    }
    fn has_offsets(&self) -> bool {
        event(&self.events, "has-offsets");
        self.c.term_offsets
    }
    fn has_payloads(&self) -> bool {
        event(&self.events, "has-payloads");
        self.c.payloads
    }
}
struct ProbeReadEnum {
    c: Rc<Case>,
    events: Events,
    next: usize,
    current: Option<usize>,
}
impl TermsEnum for ProbeReadEnum {
    fn next(&mut self) -> lucene_search::Result<Option<&[u8]>> {
        event(&self.events, "term-next");
        if self.c.next_fail {
            return Err(lucene_search::Error::IllegalArgument("next failure".into()));
        }
        self.current = (self.next < self.c.terms.len()).then_some(self.next);
        self.next += 1;
        Ok(self.term())
    }
    fn term(&self) -> Option<&[u8]> {
        self.current.map(|i| self.c.terms[i].as_slice())
    }
    fn try_seek_ceil(&mut self, _: &[u8]) -> lucene_search::Result<SeekStatus> {
        unreachable!()
    }
    fn doc_freq(&mut self) -> lucene_search::Result<i32> {
        Ok(self.c.docs.len() as i32)
    }
    fn total_term_freq(&mut self) -> lucene_search::Result<i64> {
        Ok(self.c.docs.len() as i64 * i64::from(self.c.freq))
    }
    fn postings(
        &mut self,
        _: read::PostingsFlags,
    ) -> lucene_search::Result<Box<dyn read::PostingsEnum>> {
        unreachable!()
    }
}
#[derive(Clone)]
struct ProbeProducer {
    events: Events,
    c: Rc<Case>,
    closes: Rc<RefCell<usize>>,
    fail_close: usize,
    fail_terms: bool,
    fail_integrity: bool,
    distinct: bool,
}
impl fmt::Display for ProbeProducer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ProbeProducer")
    }
}
impl FieldsProducer for ProbeProducer {
    fn field_names(&self) -> Result<Vec<String>> {
        event(&self.events, "producer-fields");
        Ok(vec!["f".into()])
    }
    fn terms(&self, name: &str) -> Result<Option<Box<dyn Terms + '_>>> {
        event(&self.events, format!("producer-terms:{name}"));
        if self.fail_terms {
            return Err(error("terms failure"));
        }
        Ok((name == "f").then(|| {
            Box::new(ProbeReadTerms {
                c: self.c.clone(),
                events: self.events.clone(),
            }) as Box<dyn Terms>
        }))
    }
    fn size(&self) -> i32 {
        event(&self.events, "producer-size");
        1
    }
    fn check_integrity(&self) -> Result<()> {
        event(&self.events, "integrity");
        if self.fail_integrity {
            Err(error("integrity failure"))
        } else {
            Ok(())
        }
    }
    fn close(&mut self) -> Result<()> {
        let mut count = self.closes.borrow_mut();
        *count += 1;
        event(&self.events, format!("pclose:{count}"));
        if *count == self.fail_close {
            Err(error("producer close failure"))
        } else {
            Ok(())
        }
    }
    fn get_merge_instance(&self) -> Self {
        event(&self.events, "merge-instance");
        let mut clone = self.clone();
        if self.distinct {
            clone.closes = Rc::default()
        }
        clone
    }
}
fn search_error(e: lucene_search::Error) -> Error {
    // Public search errors are typed; keep the fixture's original callback text.
    match e {
        lucene_search::Error::IllegalArgument(message) => error(&message),
        other => error(&other.to_string()),
    }
}
#[test]
fn actual_java_producer_wraps_terms_merge_instances_and_double_close() {
    let text = include_str!("../../../fixtures/testframework/data/asserting_postings_producer.tsv");
    for line in text.lines() {
        let row: Vec<_> = line.split('\t').collect();
        let op = row[0];
        let events = Events::default();
        let c = Case {
            next_fail: op == "next-error",
            ..Case::default()
        };
        let delegate = ProbeProducer {
            events: events.clone(),
            c: Rc::new(c),
            closes: Rc::default(),
            fail_close: if op == "close-first-error" {
                1
            } else if op == "close-second-error" {
                2
            } else {
                0
            },
            fail_terms: op == "terms-error",
            fail_integrity: op == "integrity-error",
            distinct: op == "merge-distinct",
        };
        let mut wrapper = AssertingFieldsProducer::new(delegate);
        let status = outcome(|| {
            match op {
                "fields" => event(
                    &events,
                    format!("same-fields:{}", wrapper.field_names()?[0]),
                ),
                "size" => event(&events, format!("value:{}", wrapper.size())),
                "absent" => event(
                    &events,
                    format!("absent:{}", wrapper.terms("missing")?.is_none()),
                ),
                "terms" | "terms-error" => {
                    let _ = wrapper.terms("f")?.unwrap();
                    event(&events, "wrapped:AssertingTerms")
                }
                "iterate" | "twice" | "next-error" => {
                    for _ in 0..if op == "twice" { 2 } else { 1 } {
                        let terms = wrapper.terms("f")?.unwrap();
                        let mut e = terms.iterator().map_err(search_error)?;
                        while e.next().map_err(search_error)?.is_some() {}
                    }
                }
                "stats" => {
                    let t = wrapper.terms("f")?.unwrap();
                    event(&events, format!("count:{}", t.doc_count()));
                    event(&events, format!("sum:{}", t.sum_total_term_freq()))
                }
                "integrity" | "integrity-error" => wrapper.check_integrity()?,
                "merge" | "merge-distinct" | "merge-close" => {
                    let mut merge = wrapper.get_merge_instance();
                    event(&events, "wrapped-merge:true:AssertingFieldsProducer");
                    if op == "merge-close" {
                        merge.close()?
                    } else {
                        merge
                            .terms("f")?
                            .unwrap()
                            .iterator()
                            .map_err(search_error)?
                            .next()
                            .map_err(search_error)?;
                    }
                }
                _ => wrapper.close()?,
            }
            Ok(())
        });
        assert_eq!(status, row[1], "{op}");
        assert_eq!(events.borrow().join(","), row[2], "{op}");
    }
}

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn files(prefix: &str) -> Files {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data");
    ["doc", "psm", "pos", "pay", "tim", "tip", "tmd"]
        .into_iter()
        .map(|extension| {
            (
                extension.into(),
                std::fs::read(root.join(format!("{prefix}.{extension}"))).unwrap(),
            )
        })
        .collect()
}
fn infos() -> FieldInfos {
    field_infos::parse(
        include_bytes!("../../../fixtures/testframework/data/asserting_postings.fnm"),
        &ID,
        "",
    )
    .unwrap()
}
fn read_state<'a>(files: &'a Files, infos: &'a FieldInfos, max_doc: i32) -> ReadState<'a> {
    ReadState {
        files,
        segment: SegmentReadState {
            max_doc,
            segment_id: &ID,
            segment_suffix: "s",
            field_infos: infos,
        },
    }
}
fn write_state(infos: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: infos,
    }
}
fn columns(manifest: &str, max_doc: usize) -> OwnedNorms {
    let mut norms = OwnedNorms::default();
    for line in manifest.lines() {
        let row: Vec<_> = line.split('\t').collect();
        if row[0] == "norm" {
            norms
                .0
                .entry(row[1].into())
                .or_insert_with(|| vec![None; max_doc])[row[2].parse::<usize>().unwrap()] =
                Some(row[3].parse().unwrap());
        }
    }
    norms
}
fn hex(value: &[u8]) -> String {
    if value.is_empty() {
        "-".into()
    } else {
        value.iter().map(|b| format!("{b:02x}")).collect()
    }
}
fn snapshot(
    reader: &impl FieldsProducer,
    norms: &OwnedNorms,
    infos: &FieldInfos,
    max_doc: i32,
) -> String {
    let mut result = format!("maxDoc\t{max_doc}\n");
    let mut fields: Vec<_> = infos
        .fields
        .iter()
        .filter(|field| field.index_options != IndexOptions::None)
        .collect();
    fields.sort_by(|a, b| a.name.encode_utf16().cmp(b.name.encode_utf16()));
    for field in fields {
        result += &format!(
            "field\t{}\t{}\t{}\t{}\t{}\n",
            field.name,
            field.number,
            field.index_options as u8,
            field.store_payloads,
            field.omit_norms
        );
        if let Some(values) = norms.0.get(&field.name) {
            for (doc, value) in values.iter().enumerate() {
                if let Some(value) = value {
                    result += &format!("norm\t{}\t{doc}\t{value}\n", field.name)
                }
            }
        }
        let Some(terms) = reader.terms(&field.name).unwrap() else {
            continue;
        };
        let mut e = terms.iterator().unwrap();
        while let Some(term) = e.next().unwrap() {
            let term = hex(term);
            let mut docs = e.postings(read::PostingsFlags::All).unwrap();
            loop {
                let doc = docs.next_doc().unwrap();
                if doc == NO_MORE_DOCS {
                    break;
                }
                let freq = docs.freq();
                let mut positions = Vec::new();
                let mut offsets = Vec::new();
                let mut payloads = Vec::new();
                if field.index_options.subsumes_positions() {
                    for _ in 0..freq {
                        positions.push(docs.next_position().unwrap().to_string());
                        offsets.push(format!("{}:{}", docs.start_offset(), docs.end_offset()));
                        payloads.push(hex(docs.payload().unwrap_or_default()));
                    }
                }
                let show = |values: Vec<String>| {
                    if values.is_empty() {
                        "-".into()
                    } else {
                        values.join(",")
                    }
                };
                result += &format!(
                    "posting\t{}\t{term}\t{doc}\t{freq}\t{}\t{}\t{}\n",
                    field.name,
                    show(positions),
                    show(offsets),
                    show(payloads)
                );
            }
        }
    }
    result
}
#[test]
fn actual_native_java_seven_files_values_merges_and_lifecycle() {
    let infos = infos();
    let input = files("asserting_postings");
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/testframework/data");
    let mut native = Lucene104PostingsFormat;
    let source = native
        .fields_producer(read_state(&input, &infos, 9))
        .unwrap();
    let mut asserting = AssertingPostingsFormat::new(Lucene104PostingsFormat);
    assert_eq!(asserting.name(), "Asserting");
    assert_eq!(asserting.delegate().name(), "Lucene104");
    assert_eq!(asserting.delegate_mut().name(), "Lucene104");
    for kind in 0..3 {
        let prefix = match kind {
            0 => "asserting_postings",
            1 => "asserting_postings_merge_sequential",
            _ => "asserting_postings_merge_sorted",
        };
        let max_doc = if kind == 0 { 9 } else { 16 };
        let manifest = std::fs::read_to_string(root.join(format!("{prefix}.tsv"))).unwrap();
        let norms = columns(&manifest, max_doc as usize);
        let mut writer = asserting
            .fields_consumer(write_state(&infos, max_doc))
            .unwrap();
        if kind == 0 {
            writer
                .write(&NativeFieldsView::new(&source, &infos), Some(&norms))
                .unwrap()
        } else {
            let maps: Vec<Vec<i32>> = (0..2)
                .map(|side| {
                    (0..9)
                        .map(|doc| {
                            if doc == 1 {
                                -1
                            } else {
                                let at = if doc == 0 { 0 } else { doc - 1 };
                                if kind == 2 {
                                    2 * at + side
                                } else {
                                    8 * side + at
                                }
                            }
                        })
                        .collect()
                })
                .collect();
            let sources = [
                byte::MergeSource {
                    reader: Some(&source),
                    max_doc: 9,
                    doc_map: &maps[0],
                },
                byte::MergeSource {
                    reader: Some(&source),
                    max_doc: 9,
                    doc_map: &maps[1],
                },
            ];
            writer
                .merge(
                    &MergeState {
                        bytes: byte::MergeState {
                            sources: &sources,
                            check_aborted: &|| Ok(()),
                        },
                        field_infos: &infos,
                    },
                    Some(&norms),
                )
                .unwrap();
        }
        writer.close().unwrap();
        let output = output_files(writer.delegate().output().unwrap());
        assert_eq!(output, files(prefix));
        let mut reader = asserting
            .fields_producer(read_state(&output, &infos, max_doc))
            .unwrap();
        reader.check_integrity().unwrap();
        assert_eq!(snapshot(&reader, &norms, &infos, max_doc), manifest);
        let mut expected: Vec<_> = reader.field_names().unwrap();
        expected.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        assert_eq!(reader.field_names().unwrap(), expected);
        assert_eq!(reader.size() as usize, expected.len());
        assert!(reader.terms("absent").unwrap().is_none());
        assert!(reader.to_string().starts_with("AssertingFieldsProducer("));
        let mut merge = reader.get_merge_instance();
        let terms = reader.terms("rich").unwrap().unwrap();
        merge.close().unwrap();
        assert_eq!(merge.size(), 0);
        assert!(merge.terms("rich").unwrap().is_none());
        assert!(merge.check_integrity().is_err());
        assert!(terms.iterator().unwrap().next().unwrap().is_some());
        drop(terms);
        reader.close().unwrap();
        assert!(reader.field_names().is_err());
    }
    assert_eq!(source.size(), 6);
}
