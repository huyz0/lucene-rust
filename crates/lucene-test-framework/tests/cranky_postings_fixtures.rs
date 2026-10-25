use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::postings::Position;
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::cranky_postings::*;
use lucene_util::java_random::JavaRandom;

const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const MANIFEST: &str =
    include_str!("../../../fixtures/testframework/data/cranky_postings_real.tsv");
fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "-".into()
    } else {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
fn unhex(value: &str) -> Vec<u8> {
    if value == "-" {
        return vec![];
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}
fn files(prefix: &str) -> Files {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data");
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
        include_bytes!("../../../fixtures/testframework/data/cranky_postings.fnm"),
        &ID,
        "",
    )
    .unwrap()
}
fn segment(infos: &FieldInfos, max_doc: i32) -> SegmentWriteState<'_> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: infos,
    }
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
fn model(manifest: &str) -> (OwnedFields, OwnedNorms, i32, i32) {
    let max_doc = manifest
        .lines()
        .next()
        .unwrap()
        .split('\t')
        .nth(1)
        .unwrap()
        .parse::<i32>()
        .unwrap();
    let mut fields = OwnedFields::default();
    let mut norms = OwnedNorms::default();
    let mut tail = 0;
    for line in manifest.lines().skip(1) {
        let row: Vec<_> = line.split('\t').collect();
        match row[0] {
            "field" => fields.0.push(FieldPostings {
                name: row[1].into(),
                terms: vec![],
            }),
            "norm" => {
                norms
                    .0
                    .entry(row[1].into())
                    .or_insert_with(|| vec![None; max_doc as usize])
                    [row[2].parse::<usize>().unwrap()] = Some(row[3].parse().unwrap());
            }
            "posting" => {
                let field = fields
                    .0
                    .iter_mut()
                    .find(|field| field.name == row[1])
                    .unwrap();
                let bytes = unhex(row[2]);
                if field.terms.last().is_none_or(|term| term.bytes != bytes) {
                    field.terms.push(Term {
                        bytes,
                        postings: vec![],
                    });
                }
                let positions = if row[5] == "-" {
                    vec![]
                } else {
                    let offsets: Vec<_> = row[6].split(',').collect();
                    let payloads: Vec<_> = row[7].split(',').collect();
                    row[5]
                        .split(',')
                        .enumerate()
                        .map(|(i, position)| {
                            let pair: Vec<_> = offsets[i].split(':').collect();
                            Position {
                                position: position.parse().unwrap(),
                                start_offset: pair[0].parse().unwrap(),
                                end_offset: pair[1].parse().unwrap(),
                                payload: unhex(payloads[i]),
                            }
                        })
                        .collect()
                };
                field.terms.last_mut().unwrap().postings.push(Posting {
                    doc: row[3].parse().unwrap(),
                    freq: row[4].parse().unwrap(),
                    positions,
                });
            }
            "rng" => tail = row[1].parse().unwrap(),
            _ => panic!("unknown manifest row"),
        }
    }
    (fields, norms, max_doc, tail)
}
fn snapshot(fields: &dyn Fields, norms: &OwnedNorms, infos: &FieldInfos, max_doc: i32) -> String {
    let owned = fields.materialize().unwrap();
    let mut sorted: Vec<_> = infos
        .fields
        .iter()
        .filter(|info| info.index_options != IndexOptions::None)
        .collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut result = format!("maxDoc\t{max_doc}\n");
    for info in sorted {
        result += &format!(
            "field\t{}\t{}\t{}\t{}\t{}\n",
            info.name, info.number, info.index_options as u8, info.store_payloads, info.omit_norms
        );
        if let Some(values) = norms.0.get(&info.name) {
            for (doc, value) in values.iter().enumerate() {
                if let Some(value) = value {
                    result += &format!("norm\t{}\t{doc}\t{value}\n", info.name);
                }
            }
        }
        if let Some(field) = owned.iter().find(|field| field.name == info.name) {
            for term in &field.terms {
                for posting in &term.postings {
                    let positions = if posting.positions.is_empty() {
                        "-".into()
                    } else {
                        posting
                            .positions
                            .iter()
                            .map(|p| p.position.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    let offsets = if posting.positions.is_empty() {
                        "-".into()
                    } else {
                        posting
                            .positions
                            .iter()
                            .map(|p| format!("{}:{}", p.start_offset, p.end_offset))
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    let payloads = if posting.positions.is_empty() {
                        "-".into()
                    } else {
                        posting
                            .positions
                            .iter()
                            .map(|p| hex(&p.payload))
                            .collect::<Vec<_>>()
                            .join(",")
                    };
                    result += &format!(
                        "posting\t{}\t{}\t{}\t{}\t{positions}\t{offsets}\t{payloads}\n",
                        info.name,
                        hex(&term.bytes),
                        posting.doc,
                        posting.freq
                    );
                }
            }
        }
    }
    result
}
#[derive(Default)]
struct State {
    events: Vec<String>,
    fail: bool,
    fields_identity: usize,
    norms_identity: usize,
}
type Shared = Rc<RefCell<State>>;
fn event(state: &Shared, name: impl Into<String>) -> Result<()> {
    let mut state = state.borrow_mut();
    state.events.push(name.into());
    if state.fail {
        Err(Error::Message("delegate failure".into()))
    } else {
        Ok(())
    }
}
struct RecordingProducer(Shared);
#[derive(Clone)]
struct ProducerHandle(Rc<RecordingProducer>);
impl Fields for ProducerHandle {
    fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>> {
        Ok(Cow::Borrowed(&[]))
    }
}
impl FieldsProducer for ProducerHandle {
    fn check_integrity(&self) -> Result<()> {
        event(&self.0 .0, "integrity")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0 .0, "readerClose")
    }
}
struct RecordingConsumer(Shared);
impl FieldsConsumer for RecordingConsumer {
    fn write(&mut self, fields: &dyn Fields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        let same_fields =
            fields as *const dyn Fields as *const () as usize == self.0.borrow().fields_identity;
        let same_norms = norms.is_some_and(|norms| {
            norms as *const dyn NormsProducer as *const () as usize
                == self.0.borrow().norms_identity
        });
        event(&self.0, format!("write:{same_fields}:{same_norms}"))
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0, "writerClose")
    }
    fn merge(&mut self, _: &MergeState<'_>, _: Option<&dyn NormsProducer>) -> Result<()> {
        panic!("cranky superclass merge must re-enter guarded write")
    }
}
struct RecordingFormat {
    state: Shared,
    producer: ProducerHandle,
}
impl PostingsFormat for RecordingFormat {
    type Consumer<'a> = RecordingConsumer;
    type Producer = ProducerHandle;
    fn name(&self) -> &str {
        "RecordingPostings"
    }
    fn fields_consumer<'a>(&mut self, _: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        event(&self.state, "consumer")?;
        Ok(RecordingConsumer(self.state.clone()))
    }
    fn fields_producer(&mut self, _: ReadState<'_>) -> Result<Self::Producer> {
        event(&self.state, "producer")?;
        Ok(self.producer.clone())
    }
}
#[test]
fn actual_java_wrapper_preserves_callback_identity_errors_superclass_merge_and_rng() {
    let infos = infos();
    let files = files("cranky_postings");
    let mut expected =
        include_str!("../../../fixtures/testframework/data/cranky_postings.tsv").lines();
    for seed in [0, 1, 42, -1, i64::MIN, i64::MAX] {
        let state = Rc::new(RefCell::new(State::default()));
        let mut source = Box::new(ProducerHandle(Rc::new(RecordingProducer(state.clone()))));
        let norms = Box::new(OwnedNorms::default());
        state.borrow_mut().fields_identity = &*source as *const ProducerHandle as usize;
        state.borrow_mut().norms_identity = &*norms as *const OwnedNorms as usize;
        let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
        let mut format = CrankyPostingsFormat::new(
            RecordingFormat {
                state: state.clone(),
                producer: source.as_ref().clone(),
            },
            random.clone(),
        );
        let mut consumer = loop {
            if let Ok(consumer) = format.fields_consumer(segment(&infos, 4)) {
                break consumer;
            }
        };
        let sources = [
            MergeSource {
                reader: Some(&*source),
                max_doc: 2,
                doc_map: &[0, 1],
            },
            MergeSource {
                reader: None,
                max_doc: 0,
                doc_map: &[],
            },
            MergeSource {
                reader: Some(&*source),
                max_doc: 2,
                doc_map: &[2, 3],
            },
        ];
        let merge = MergeState {
            sources: &sources,
            check_aborted: &|| Ok(()),
        };
        for step in 0..2400 {
            state.borrow_mut().events.clear();
            state.borrow_mut().fail = step % 11 == 0;
            let op = step % 8;
            let result = match op {
                0 => Ok(format.name().to_string()),
                1 => format
                    .fields_consumer(segment(&infos, 4))
                    .map(|_| "ok".into()),
                2 => consumer.write(&*source, Some(&*norms)).map(|_| "ok".into()),
                3 => consumer.close().map(|_| "ok".into()),
                4 => format
                    .fields_producer(read_state(&files, &infos, 9))
                    .map(|producer| Rc::ptr_eq(&producer.0, &source.0).to_string()),
                5 => source.check_integrity().map(|_| "ok".into()),
                6 => {
                    let mut clone = source.as_ref().clone();
                    clone.close().map(|_| "ok".into())
                }
                7 => consumer.merge(&merge, Some(&*norms)).map(|_| "ok".into()),
                _ => unreachable!(),
            };
            let status = result.unwrap_or_else(|error| error.to_string());
            let events = state.borrow().events.join(",");
            let actual = format!(
                "{seed}\t{step}\t{op}\t{status}\t{events}\t{}",
                random.borrow_mut().next_int()
            );
            assert_eq!(
                expected.next(),
                Some(actual.as_str()),
                "seed={seed} step={step}"
            );
        }
        assert_eq!(format.delegate().name(), "RecordingPostings");
        assert!(Rc::ptr_eq(&format.delegate_mut().producer.0, &source.0));
        source.close().ok();
    }
    assert!(expected.next().is_none());
}
#[test]
fn real_lucene104_reads_and_writes_every_java_byte_and_all_posting_features() {
    let infos = infos();
    let files = files("cranky_postings");
    let (input, norms, max_doc, tail) = model(MANIFEST);
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut format = CrankyPostingsFormat::new(Lucene104PostingsFormat, random.clone());
    let mut reader = format
        .fields_producer(read_state(&files, &infos, max_doc))
        .unwrap();
    reader.check_integrity().unwrap();
    let expected = MANIFEST
        .lines()
        .filter(|line| !line.starts_with("rng\t"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    assert_eq!(snapshot(&reader, &norms, &infos, max_doc), expected);
    let mut writer = format.fields_consumer(segment(&infos, max_doc)).unwrap();
    writer.write(&input, Some(&norms)).unwrap();
    writer.close().unwrap();
    assert_eq!(output_files(writer.delegate().output().unwrap()), files);
    assert_eq!(random.borrow_mut().next_int(), tail);
    assert!(writer.delegate().output().is_some());
    assert!(writer.delegate_mut().write(&input, Some(&norms)).is_err());
    reader.close().unwrap();
    reader.close().unwrap();
    assert!(reader.materialize().is_err());
    assert!(reader.check_integrity().is_err());
}
#[test]
fn actual_superclass_mapped_merges_match_sequential_and_sorted_java_bytes() {
    let infos = infos();
    let input_files = files("cranky_postings");
    let mut native = Lucene104PostingsFormat;
    let reader = native
        .fields_producer(read_state(&input_files, &infos, 9))
        .unwrap();
    for sorted in [false, true] {
        let prefix = if sorted {
            "cranky_postings_merge_sorted"
        } else {
            "cranky_postings_merge_sequential"
        };
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data");
        let manifest = std::fs::read_to_string(root.join(format!("{prefix}.tsv"))).unwrap();
        let (_, norms, max_doc, tail) = model(&manifest);
        let maps: Vec<Vec<i32>> = (0..2)
            .map(|side| {
                (0..9)
                    .map(|doc| {
                        if doc == 1 {
                            -1
                        } else {
                            let id = if doc == 0 { 0 } else { doc - 1 };
                            if sorted {
                                2 * id + side
                            } else {
                                8 * side + id
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        let sources = [
            MergeSource {
                reader: Some(&reader),
                max_doc: 9,
                doc_map: &maps[0],
            },
            MergeSource {
                reader: Some(&reader),
                max_doc: 9,
                doc_map: &maps[1],
            },
        ];
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut format = CrankyPostingsFormat::new(Lucene104PostingsFormat, random.clone());
        let mut consumer = format.fields_consumer(segment(&infos, max_doc)).unwrap();
        consumer
            .merge(
                &MergeState {
                    sources: &sources,
                    check_aborted: &|| Ok(()),
                },
                Some(&norms),
            )
            .unwrap();
        consumer.close().unwrap();
        let actual = output_files(consumer.delegate().output().unwrap());
        assert_eq!(actual, files(prefix));
        assert_eq!(random.borrow_mut().next_int(), tail);
        let merged = native
            .fields_producer(read_state(&actual, &infos, max_doc))
            .unwrap();
        merged.check_integrity().unwrap();
        let expected = manifest
            .lines()
            .filter(|line| !line.starts_with("rng\t"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert_eq!(snapshot(&merged, &norms, &infos, max_doc), expected);
        assert!(!expected.contains("676f6e65"));
    }
}

// Record the actual Fields handed to guarded write before the native consumer
// traverses it, just like the Java generator's recording PostingsFormat.
struct OrderFormat(Rc<RefCell<Vec<String>>>);
struct OrderConsumer<W> {
    delegate: W,
    observed: Rc<RefCell<Vec<String>>>,
}
impl<W: FieldsConsumer> FieldsConsumer for OrderConsumer<W> {
    fn write(&mut self, fields: &dyn Fields, norms: Option<&dyn NormsProducer>) -> Result<()> {
        *self.observed.borrow_mut() = fields
            .materialize()?
            .iter()
            .map(|field| field.name.clone())
            .collect();
        self.delegate.write(fields, norms)
    }
    fn close(&mut self) -> Result<()> {
        self.delegate.close()
    }
}
impl PostingsFormat for OrderFormat {
    type Consumer<'a> = OrderConsumer<Lucene104FieldsConsumer<'a>>;
    type Producer = Lucene104FieldsProducer;
    fn name(&self) -> &str {
        "Lucene104"
    }
    fn fields_consumer<'a>(&mut self, state: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        Ok(OrderConsumer {
            delegate: Lucene104PostingsFormat.fields_consumer(state)?,
            observed: self.0.clone(),
        })
    }
    fn fields_producer(&mut self, state: ReadState<'_>) -> Result<Self::Producer> {
        Lucene104PostingsFormat.fields_producer(state)
    }
}
fn unicode_infos() -> FieldInfos {
    field_infos::parse(
        include_bytes!("../../../fixtures/testframework/data/cranky_postings_unicode.fnm"),
        &ID,
        "",
    )
    .unwrap()
}
fn unicode_order(prefix: &str) -> (Vec<String>, i32) {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data");
    let manifest = std::fs::read_to_string(root.join(format!("{prefix}_order.tsv"))).unwrap();
    let mut names = Vec::new();
    let mut tail = None;
    for line in manifest.lines() {
        let (kind, value) = line.split_once('\t').unwrap();
        match kind {
            "field" => names.push(value.to_string()),
            "rng" => tail = Some(value.parse().unwrap()),
            _ => panic!("unexpected order row"),
        }
    }
    assert_eq!(names.len(), 2);
    (names, tail.unwrap())
}
#[test]
fn unicode_native_reader_field_names_match_actual_java_utf16_order() {
    let infos = unicode_infos();
    let input = files("cranky_postings_unicode");
    let source = Lucene104PostingsFormat
        .fields_producer(read_state(&input, &infos, 9))
        .unwrap();
    source.check_integrity().unwrap();
    let names: Vec<_> = source
        .materialize()
        .unwrap()
        .iter()
        .map(|field| field.name.clone())
        .collect();
    assert_eq!(names, unicode_order("cranky_postings_unicode").0);
}
#[test]
fn unicode_ordinary_write_matches_java_field_callback_order_and_all_seven_files() {
    let infos = unicode_infos();
    let input = files("cranky_postings_unicode");
    let source = Lucene104PostingsFormat
        .fields_producer(read_state(&input, &infos, 9))
        .unwrap();
    let observed = Rc::new(RefCell::new(Vec::new()));
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut format = CrankyPostingsFormat::new(OrderFormat(observed.clone()), random.clone());
    let mut writer = format.fields_consumer(segment(&infos, 9)).unwrap();
    writer.write(&source, None).unwrap();
    writer.close().unwrap();
    let (expected, tail) = unicode_order("cranky_postings_unicode");
    assert_eq!(*observed.borrow(), expected);
    assert_eq!(
        output_files(writer.delegate().delegate.output().unwrap()),
        input
    );
    assert_eq!(random.borrow_mut().next_int(), tail);
}
fn unicode_merge(sorted: bool) {
    let infos = unicode_infos();
    let input = files("cranky_postings_unicode");
    let source = Lucene104PostingsFormat
        .fields_producer(read_state(&input, &infos, 9))
        .unwrap();
    let prefix = if sorted {
        "cranky_postings_unicode_merge_sorted"
    } else {
        "cranky_postings_unicode_merge_sequential"
    };
    let maps: Vec<Vec<i32>> = (0..2)
        .map(|side| {
            (0..9)
                .map(|doc| {
                    if doc == 1 {
                        -1
                    } else {
                        let at = if doc == 0 { 0 } else { doc - 1 };
                        if sorted {
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
        MergeSource {
            reader: Some(&source),
            max_doc: 9,
            doc_map: &maps[0],
        },
        MergeSource {
            reader: Some(&source),
            max_doc: 9,
            doc_map: &maps[1],
        },
    ];
    let observed = Rc::new(RefCell::new(Vec::new()));
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut format = CrankyPostingsFormat::new(OrderFormat(observed.clone()), random.clone());
    let mut writer = format.fields_consumer(segment(&infos, 16)).unwrap();
    writer
        .merge(
            &MergeState {
                sources: &sources,
                check_aborted: &|| Ok(()),
            },
            None,
        )
        .unwrap();
    writer.close().unwrap();
    let (expected, tail) = unicode_order(prefix);
    assert_eq!(*observed.borrow(), expected);
    let output = output_files(writer.delegate().delegate.output().unwrap());
    assert_eq!(output, files(prefix));
    assert_eq!(random.borrow_mut().next_int(), tail);
    let merged = Lucene104PostingsFormat
        .fields_producer(read_state(&output, &infos, 16))
        .unwrap();
    merged.check_integrity().unwrap();
    assert_eq!(
        merged
            .materialize()
            .unwrap()
            .iter()
            .map(|field| field.name.clone())
            .collect::<Vec<_>>(),
        expected
    );
    assert!(merged
        .materialize()
        .unwrap()
        .iter()
        .all(|field| field.terms.iter().all(|term| term.bytes != b"gone")));
}
#[test]
fn unicode_sequential_merge_matches_java_mapped_order_and_all_seven_files() {
    unicode_merge(false);
}
#[test]
fn unicode_sorted_merge_matches_java_mapped_order_and_all_seven_files() {
    unicode_merge(true);
}
