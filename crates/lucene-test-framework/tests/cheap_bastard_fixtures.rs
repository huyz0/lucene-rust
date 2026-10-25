use lucene_codecs::field_infos::{self, FieldInfos, IndexOptions};
use lucene_codecs::postings::Position;
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::cheap_bastard::*;
use lucene_test_framework::codecs::cranky_postings::{
    FieldPostings, Fields, FieldsConsumer, FieldsProducer, MergeSource, MergeState, Posting,
    PostingsFormat, Term,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const MANIFEST: &str = include_str!("../../../fixtures/testframework/data/cheap_bastard.tsv");
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
        include_bytes!("../../../fixtures/testframework/data/cheap_bastard.fnm"),
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

fn manifest(prefix: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../fixtures/testframework/data/{prefix}.tsv")),
    )
    .unwrap()
}
fn assert_seeks(bytes: &Files, infos: &FieldInfos, prefix: &str, max_doc: i32) {
    use lucene_codecs::blocktree::{self, SeekStatus};
    let fields = blocktree::open(
        &bytes["tim"],
        &bytes["tip"],
        &bytes["tmd"],
        infos,
        &ID,
        "s",
        max_doc,
    )
    .unwrap();
    for line in manifest(&format!("{prefix}_seeks")).lines() {
        let row: Vec<_> = line.split('\t').collect();
        let field = fields
            .iter_fields()
            .find(|(name, _)| *name == row[0])
            .unwrap()
            .1;
        let mut terms = field.iter();
        let status = terms.try_seek_ceil(row[1].as_bytes()).unwrap();
        assert_eq!(
            status,
            match row[2] {
                "FOUND" => SeekStatus::Found,
                "NOT_FOUND" => SeekStatus::NotFound,
                "END" => SeekStatus::End,
                _ => panic!("unknown seek status"),
            }
        );
        if status != SeekStatus::End {
            assert_eq!(terms.term().unwrap(), row[3].as_bytes());
            let stats = terms.try_stats().unwrap().unwrap();
            assert_eq!(stats.doc_freq, row[4].parse::<i32>().unwrap());
            assert_eq!(stats.total_term_freq, row[5].parse::<i64>().unwrap());
        }
    }
}
#[test]
fn actual_java_component_reads_writes_and_seeks_all_selected_bytes() {
    let infos = infos();
    let bytes = files("cheap_bastard");
    let (input, norms, max_doc, _) = model(MANIFEST);
    let mut codec = CheapBastardCodec::new();
    let mut reader = codec
        .postings_format_mut()
        .fields_producer(read_state(&bytes, &infos, max_doc))
        .unwrap();
    reader.check_integrity().unwrap();
    assert_eq!(snapshot(&reader, &norms, &infos, max_doc), MANIFEST);
    assert_eq!(
        input
            .0
            .iter()
            .find(|f| f.name == "rich")
            .unwrap()
            .terms
            .len(),
        1536
    );
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, max_doc))
        .unwrap();
    writer.write(&input, Some(&norms)).unwrap();
    assert!(writer.write(&input, Some(&norms)).is_err());
    writer.close().unwrap();
    writer.close().unwrap();
    assert_eq!(output_files(writer.output().unwrap()), bytes);
    assert!(writer.write(&input, Some(&norms)).is_err());
    assert_seeks(&bytes, &infos, "cheap_bastard", max_doc);
    let defaults = files("cheap_bastard_default");
    assert_ne!(bytes["tim"], defaults["tim"]);
    assert_ne!(bytes["tip"], defaults["tip"]);
    let default_reader = codec
        .postings_format_mut()
        .fields_producer(read_state(&defaults, &infos, max_doc))
        .unwrap();
    assert_eq!(snapshot(&default_reader, &norms, &infos, max_doc), MANIFEST);
    assert_seeks(&defaults, &infos, "cheap_bastard_default", max_doc);
    reader.close().unwrap();
    assert!(reader.materialize().is_err());
}
#[test]
fn sequential_and_sorted_merges_match_java_all_seven_files_and_seeks() {
    let infos = infos();
    let input = files("cheap_bastard");
    let mut codec = CheapBastardCodec::new();
    let source = codec
        .postings_format_mut()
        .fields_producer(read_state(&input, &infos, 9))
        .unwrap();
    for sorted in [false, true] {
        let prefix = if sorted {
            "cheap_bastard_merge_sorted"
        } else {
            "cheap_bastard_merge_sequential"
        };
        let expected = manifest(prefix);
        let (_, norms, max_doc, _) = model(&expected);
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
        let mut writer = codec
            .postings_format_mut()
            .fields_consumer(segment(&infos, max_doc))
            .unwrap();
        writer
            .merge(
                &MergeState {
                    sources: &sources,
                    check_aborted: &|| Ok(()),
                },
                Some(&norms),
            )
            .unwrap();
        writer.close().unwrap();
        let actual = output_files(writer.output().unwrap());
        assert_eq!(actual, files(prefix));
        let reader = codec
            .postings_format_mut()
            .fields_producer(read_state(&actual, &infos, max_doc))
            .unwrap();
        reader.check_integrity().unwrap();
        assert_eq!(snapshot(&reader, &norms, &infos, max_doc), expected);
        assert_seeks(&actual, &infos, prefix, max_doc);
    }
}

#[test]
fn actual_configuration_and_every_per_invocation_delegate_callback() {
    use lucene_codecs::codec::Lucene104Codec;
    let codec = CheapBastardCodec::default();
    let actual = manifest("cheap_bastard_config");
    assert!(actual.contains(&format!("name\t{}\n", codec.name())));
    assert!(actual.contains(&format!("delegate\t{}\n", codec.delegate().name())));
    let format = codec.postings_format();
    let settings = format.settings();
    assert_eq!(
        (settings.min_term_block_size, settings.max_term_block_size),
        (100, 200)
    );
    assert!(actual.contains(&format!("postings\t{}\n", format.name())));
    assert!(std::ptr::eq(format, codec.postings_format()));
    struct Delegate {
        calls: RefCell<Vec<ComponentKind>>,
        stable: Rc<()>,
        fail: Cell<bool>,
    }
    impl CodecDelegate for Delegate {
        type Component = Rc<()>;
        fn component(&self, kind: ComponentKind) -> Result<Rc<()>> {
            self.calls.borrow_mut().push(kind);
            if self.fail.get() {
                return Err(Error::Message("first delegate error".into()));
            }
            Ok(if kind == ComponentKind::Points {
                Rc::new(())
            } else {
                self.stable.clone()
            })
        }
    }
    let codec = CheapBastardCodec::with_delegate(Delegate {
        calls: RefCell::new(vec![]),
        stable: Rc::new(()),
        fail: Cell::new(false),
    });
    use ComponentKind::*;
    for (kind, accessor) in [
        (DocValues, "docValuesFormat"),
        (FieldInfos, "fieldInfosFormat"),
        (LiveDocs, "liveDocsFormat"),
        (Norms, "normsFormat"),
        (SegmentInfo, "segmentInfoFormat"),
        (StoredFields, "storedFieldsFormat"),
        (TermVectors, "termVectorsFormat"),
        (Compound, "compoundFormat"),
        (Points, "pointsFormat"),
        (KnnVectors, "knnVectorsFormat"),
    ] {
        let before = codec.delegate().calls.borrow().len();
        let first = codec.component(kind).unwrap();
        let second = codec.component(kind).unwrap();
        let expected = actual
            .lines()
            .find(|line| line.starts_with(&format!("delegate-identity\t{accessor}\t")))
            .unwrap();
        assert_eq!(Rc::ptr_eq(&first, &second), expected.ends_with("\ttrue"));
        assert_eq!(&codec.delegate().calls.borrow()[before..], &[kind, kind]);
        codec.delegate().fail.set(true);
        assert_eq!(
            codec.component(kind).unwrap_err().to_string(),
            "first delegate error"
        );
        codec.delegate().fail.set(false);
    }
}

#[test]
fn component_writer_validates_boundaries_and_propagates_sources() {
    use std::borrow::Cow;
    let mut infos = infos();
    let (mut input, _, _, _) = model(MANIFEST);
    let mut codec = CheapBastardCodec::new();
    assert!(codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, -1))
        .is_err());
    let mut empty = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 0))
        .unwrap();
    empty.close().unwrap();
    assert!(empty.output().is_some());
    empty.close().unwrap();
    struct BadFields;
    impl Fields for BadFields {
        fn materialize(&self) -> Result<Cow<'_, [FieldPostings]>> {
            Err(Error::Message("source fields error".into()))
        }
    }
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 9))
        .unwrap();
    assert_eq!(
        writer.write(&BadFields, None).unwrap_err().to_string(),
        "source fields error"
    );
    let saved = input.0[0].name.clone();
    input.0[0].name = "unknown".into();
    assert!(writer
        .write(&input, None)
        .unwrap_err()
        .to_string()
        .contains("unrecognized"));
    input.0[0].name = saved;
    input.0.retain(|field| !field.terms.is_empty());
    let old = input.0[0].terms[0].postings[0].doc;
    for doc in [-1, 9] {
        input.0[0].terms[0].postings[0].doc = doc;
        assert!(writer
            .write(&input, None)
            .unwrap_err()
            .to_string()
            .contains("out of bounds"));
    }
    input.0[0].terms[0].postings[0].doc = old;
    struct BadNorms;
    impl NormsProducer for BadNorms {
        fn norms(&self, _: &field_infos::FieldInfo) -> Result<Option<&[Option<i64>]>> {
            Err(Error::Message("source norms error".into()))
        }
    }
    assert_eq!(
        writer
            .write(&input, Some(&BadNorms))
            .unwrap_err()
            .to_string(),
        "source norms error"
    );
    // Absent columns and absent norm values follow the shared native adapter.
    input.0[0].terms.push(Term {
        bytes: b"zzzz".to_vec(),
        postings: vec![],
    });
    writer.write(&input, Some(&OwnedNorms::default())).unwrap();
    writer.close().unwrap();
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 9))
        .unwrap();
    writer
        .write(
            &input,
            Some(&OwnedNorms(std::collections::BTreeMap::from([(
                "rich".into(),
                vec![None; 9],
            )]))),
        )
        .unwrap();
    let name = input.0[0].name.clone();
    infos
        .fields
        .iter_mut()
        .find(|field| field.name == name)
        .unwrap()
        .index_options = IndexOptions::None;
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 9))
        .unwrap();
    assert!(writer
        .write(&input, None)
        .unwrap_err()
        .to_string()
        .contains("does not index"));
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 0))
        .unwrap();
    writer.write(&OwnedFields::default(), None).unwrap();
    writer.close().unwrap();
}

#[test]
fn selected_reader_refuses_missing_corrupt_and_wrong_segment_files() {
    let infos = infos();
    let original = files("cheap_bastard");
    let mut codec = CheapBastardCodec::new();
    for ext in ["doc", "psm", "pos", "pay", "tim", "tip", "tmd"] {
        let mut corrupt = original.clone();
        corrupt.get_mut(ext).unwrap()[0] ^= 1;
        assert!(
            codec
                .postings_format_mut()
                .fields_producer(read_state(&corrupt, &infos, 9))
                .is_err(),
            "{ext}"
        );
        let mut truncated = original.clone();
        truncated.get_mut(ext).unwrap().truncate(5);
        assert!(
            codec
                .postings_format_mut()
                .fields_producer(read_state(&truncated, &infos, 9))
                .is_err(),
            "{ext}"
        );
        let mut missing = original.clone();
        missing.remove(ext);
        assert!(
            codec
                .postings_format_mut()
                .fields_producer(read_state(&missing, &infos, 9))
                .is_err(),
            "{ext}"
        );
    }
    let wrong = [99; 16];
    let mut state = read_state(&original, &infos, 9);
    state.segment.segment_id = &wrong;
    assert!(codec.postings_format_mut().fields_producer(state).is_err());
    let mut state = read_state(&original, &infos, 9);
    state.segment.segment_suffix = "other";
    assert!(codec.postings_format_mut().fields_producer(state).is_err());
}

struct Source {
    fields: Vec<FieldPostings>,
    owned: bool,
    name: &'static str,
    events: Rc<RefCell<Vec<String>>>,
    fail_integrity: bool,
    fail_fields: bool,
}
impl Fields for Source {
    fn materialize(&self) -> Result<std::borrow::Cow<'_, [FieldPostings]>> {
        self.events
            .borrow_mut()
            .push(format!("fields:{}", self.name));
        if self.fail_fields {
            return Err(Error::Message(format!("fields:{}", self.name)));
        }
        Ok(if self.owned {
            std::borrow::Cow::Owned(self.fields.clone())
        } else {
            std::borrow::Cow::Borrowed(&self.fields)
        })
    }
}
impl FieldsProducer for Source {
    fn check_integrity(&self) -> Result<()> {
        self.events
            .borrow_mut()
            .push(format!("integrity:{}", self.name));
        if self.fail_integrity {
            Err(Error::Message(format!("integrity:{}", self.name)))
        } else {
            Ok(())
        }
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn source(
    fields: Vec<FieldPostings>,
    name: &'static str,
    events: &Rc<RefCell<Vec<String>>>,
) -> Source {
    Source {
        fields,
        owned: false,
        name,
        events: events.clone(),
        fail_integrity: false,
        fail_fields: false,
    }
}
fn one_posting() -> Vec<FieldPostings> {
    vec![FieldPostings {
        name: "docs".into(),
        terms: vec![Term {
            bytes: b"term".to_vec(),
            postings: vec![Posting {
                doc: 0,
                freq: 1,
                positions: vec![],
            }],
        }],
    }]
}
#[test]
fn merged_owned_source_lifetime_and_null_source_match_every_java_byte() {
    let infos = infos();
    let (input, norms, _, _) = model(MANIFEST);
    let events = Rc::new(RefCell::new(vec![]));
    let mut source = source(input.0, "owned", &events);
    source.owned = true;
    let map: Vec<_> = (0..9).collect();
    let sources = [
        MergeSource {
            reader: None,
            max_doc: -1,
            doc_map: &[],
        },
        MergeSource {
            reader: Some(&source),
            max_doc: 9,
            doc_map: &map,
        },
    ];
    let abort = || {
        events.borrow_mut().push("abort".into());
        Ok(())
    };
    let mut codec = CheapBastardCodec::new();
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 9))
        .unwrap();
    writer
        .merge(
            &MergeState {
                sources: &sources,
                check_aborted: &abort,
            },
            Some(&norms),
        )
        .unwrap();
    writer.close().unwrap();
    assert_eq!(
        output_files(writer.output().unwrap()),
        files("cheap_bastard")
    );
    assert_eq!(
        *events.borrow(),
        ["abort", "integrity:owned", "fields:owned"]
    );
}
#[test]
fn mapped_merge_preserves_source_callback_and_first_error_order() {
    let infos = infos();
    let events = Rc::new(RefCell::new(vec![]));
    let mut codec = CheapBastardCodec::new();
    for (doc, map, error, materialize_second) in [
        (0, vec![], "document map out of bounds", false),
        (0, vec![-2], "invalid mapped document", false),
        (-1, vec![0], "source document out of bounds", false),
        (1, vec![0], "source document out of bounds", false),
        (0, vec![9], "destination document out of bounds", true),
    ] {
        events.borrow_mut().clear();
        let mut first = source(one_posting(), "first", &events);
        first.fields[0].terms[0].postings[0].doc = doc;
        let second = source(one_posting(), "second", &events);
        let sources = [
            MergeSource {
                reader: Some(&first),
                max_doc: 1,
                doc_map: &map,
            },
            MergeSource {
                reader: Some(&second),
                max_doc: 1,
                doc_map: &[0],
            },
        ];
        let abort = || {
            events.borrow_mut().push("abort".into());
            Ok(())
        };
        let mut writer = codec
            .postings_format_mut()
            .fields_consumer(segment(&infos, 1))
            .unwrap();
        assert_eq!(
            writer
                .merge(
                    &MergeState {
                        sources: &sources,
                        check_aborted: &abort
                    },
                    None
                )
                .unwrap_err()
                .to_string(),
            error
        );
        let mut expected = vec![
            "abort",
            "integrity:first",
            "abort",
            "integrity:second",
            "fields:first",
        ];
        if materialize_second {
            expected.push("fields:second");
        }
        assert_eq!(*events.borrow(), expected);
    }
    let mut first = source(one_posting(), "first", &events);
    let second = source(one_posting(), "second", &events);
    first.fail_fields = true;
    let abort = || {
        events.borrow_mut().push("abort".into());
        Ok(())
    };
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 1))
        .unwrap();
    for failure in ["fields", "integrity", "abort", "closed", "written"] {
        events.borrow_mut().clear();
        first.fail_fields = failure == "fields";
        first.fail_integrity = failure == "integrity";
        let sources = [
            MergeSource {
                reader: Some(&first),
                max_doc: 1,
                doc_map: &[0],
            },
            MergeSource {
                reader: Some(&second),
                max_doc: 1,
                doc_map: &[0],
            },
        ];
        let abort_fail = || {
            events.borrow_mut().push("abort".into());
            Err(Error::Message("abort failed".into()))
        };
        let check: &dyn Fn() -> Result<()> = if failure == "abort" {
            &abort_fail
        } else {
            &abort
        };
        if failure == "closed" {
            writer.close().unwrap();
        }
        if failure == "written" {
            writer = codec
                .postings_format_mut()
                .fields_consumer(segment(&infos, 1))
                .unwrap();
            writer.write(&OwnedFields::default(), None).unwrap();
        }
        assert!(writer
            .merge(
                &MergeState {
                    sources: &sources,
                    check_aborted: check
                },
                None
            )
            .is_err());
        let expected = match failure {
            "abort" => vec!["abort"],
            "integrity" => vec!["abort", "integrity:first"],
            "fields" => vec![
                "abort",
                "integrity:first",
                "abort",
                "integrity:second",
                "fields:first",
            ],
            _ => vec!["abort", "integrity:first", "abort", "integrity:second"],
        };
        assert_eq!(*events.borrow(), expected);
    }
}
#[test]
fn merge_deletions_empty_terms_unicode_norm_order_and_encoder_failures() {
    let mut infos = infos();
    let events = Rc::new(RefCell::new(vec![]));
    let mut codec = CheapBastardCodec::new();
    let source = source(one_posting(), "deleted", &events);
    let sources = [MergeSource {
        reader: Some(&source),
        max_doc: 1,
        doc_map: &[-1],
    }];
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 0))
        .unwrap();
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
    let mut expected = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 0))
        .unwrap();
    expected.close().unwrap();
    assert_eq!(
        output_files(writer.output().unwrap()),
        output_files(expected.output().unwrap())
    );
    let mut unknown = one_posting();
    unknown[0].name = "unknown".into();
    let unknown = Source {
        fields: unknown,
        ..source
    };
    let sources = [MergeSource {
        reader: Some(&unknown),
        max_doc: 1,
        doc_map: &[-1],
    }];
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 0))
        .unwrap();
    assert!(writer
        .merge(
            &MergeState {
                sources: &sources,
                check_aborted: &|| Ok(())
            },
            None
        )
        .unwrap_err()
        .to_string()
        .contains("unrecognized"));
    let actual_order: Vec<_> = manifest("cranky_postings_unicode_order")
        .lines()
        .filter_map(|line| line.strip_prefix("field\t").map(str::to_owned))
        .collect();
    let names = ["\u{e000}", "\u{10000}"];
    let template = infos
        .fields
        .iter()
        .find(|info| info.name == "docs")
        .unwrap()
        .clone();
    infos.fields = names
        .iter()
        .enumerate()
        .map(|(number, name)| {
            let mut info = template.clone();
            info.name = (*name).into();
            info.number = number as i32;
            info.omit_norms = false;
            info
        })
        .collect();
    let fields = names
        .iter()
        .map(|name| {
            let mut fields = one_posting();
            fields[0].name = (*name).into();
            fields.pop().unwrap()
        })
        .collect();
    let source = Source {
        fields,
        owned: false,
        name: "unicode",
        events: events.clone(),
        fail_fields: false,
        fail_integrity: false,
    };
    struct NormOrder(RefCell<Vec<String>>);
    impl NormsProducer for NormOrder {
        fn norms(&self, info: &field_infos::FieldInfo) -> Result<Option<&[Option<i64>]>> {
            self.0.borrow_mut().push(info.name.clone());
            Ok(None)
        }
    }
    let norms = NormOrder(RefCell::new(vec![]));
    let sources = [MergeSource {
        reader: Some(&source),
        max_doc: 1,
        doc_map: &[0],
    }];
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 1))
        .unwrap();
    writer
        .merge(
            &MergeState {
                sources: &sources,
                check_aborted: &|| Ok(()),
            },
            Some(&norms),
        )
        .unwrap();
    assert_eq!(*norms.0.borrow(), actual_order);
    // A duplicate mapped document reaches the real encoder's validation.
    let sources = [
        MergeSource {
            reader: Some(&source),
            max_doc: 1,
            doc_map: &[0],
        },
        MergeSource {
            reader: Some(&source),
            max_doc: 1,
            doc_map: &[0],
        },
    ];
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 1))
        .unwrap();
    assert!(writer
        .merge(
            &MergeState {
                sources: &sources,
                check_aborted: &|| Ok(())
            },
            None
        )
        .is_err());
    assert!(writer.output().is_none());
}

#[test]
fn merge_groups_repeated_unsorted_terms_and_fields_and_deleted_ranges() {
    let infos = infos();
    let events = Rc::new(RefCell::new(vec![]));
    let term = |bytes: &[u8], doc| Term {
        bytes: bytes.to_vec(),
        postings: vec![Posting {
            doc,
            freq: 1,
            positions: vec![],
        }],
    };
    let mut first = source(
        vec![
            FieldPostings {
                name: "docs".into(),
                terms: vec![
                    term(b"z", 2),
                    term(b"a", 0),
                    term(b"a", 1),
                    term(b"gone", 3),
                ],
            },
            FieldPostings {
                name: "docs".into(),
                terms: vec![term(b"z", 0)],
            },
        ],
        "first",
        &events,
    );
    first.owned = true;
    let second = source(
        vec![FieldPostings {
            name: "docs".into(),
            terms: vec![term(b"b", 0)],
        }],
        "second",
        &events,
    );
    let sources = [
        MergeSource {
            reader: Some(&first),
            max_doc: 4,
            doc_map: &[0, 1, 2, -1],
        },
        MergeSource {
            reader: Some(&second),
            max_doc: 1,
            doc_map: &[1],
        },
    ];
    let mut codec = CheapBastardCodec::new();
    let mut writer = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 3))
        .unwrap();
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
    let mut a = term(b"a", 0);
    a.postings.push(term(b"a", 1).postings.pop().unwrap());
    let mut z = term(b"z", 0);
    z.postings.push(term(b"z", 2).postings.pop().unwrap());
    let expected = OwnedFields(vec![FieldPostings {
        name: "docs".into(),
        terms: vec![a, term(b"b", 1), z],
    }]);
    let mut direct = codec
        .postings_format_mut()
        .fields_consumer(segment(&infos, 3))
        .unwrap();
    direct.write(&expected, None).unwrap();
    direct.close().unwrap();
    assert_eq!(
        output_files(writer.output().unwrap()),
        output_files(direct.output().unwrap())
    );
    assert_eq!(
        *events.borrow(),
        [
            "integrity:first",
            "integrity:second",
            "fields:first",
            "fields:second"
        ]
    );
}
