#![cfg(test)]
use super::*;
use lucene_codecs::term_vectors::TermVectorTerm;
use lucene_store::{ByteBuffersDirectory, SliceInput};

#[derive(Default)]
struct Probe {
    positions: Vec<(i32, i32, i32, Option<Vec<u8>>)>,
    calls: Vec<&'static str>,
}
impl TermVectorsWriter for Probe {
    fn start_document(&mut self, _: i32) -> io::Result<()> {
        self.calls.push("doc");
        Ok(())
    }
    fn finish_document(&mut self) -> io::Result<()> {
        self.calls.push("enddoc");
        Ok(())
    }
    fn start_field(&mut self, _: &FieldInfo, _: i32, _: bool, _: bool, _: bool) -> io::Result<()> {
        self.calls.push("field");
        Ok(())
    }
    fn finish_field(&mut self) -> io::Result<()> {
        self.calls.push("endfield");
        Ok(())
    }
    fn start_term(&mut self, _: &[u8], _: i32) -> io::Result<()> {
        self.calls.push("term");
        Ok(())
    }
    fn finish_term(&mut self) -> io::Result<()> {
        self.calls.push("endterm");
        Ok(())
    }
    fn add_position(&mut self, p: i32, s: i32, e: i32, y: Option<&[u8]>) -> io::Result<()> {
        self.positions.push((p, s, e, y.map(<[u8]>::to_vec)));
        Ok(())
    }
    fn finish(&mut self, _: i32) -> io::Result<()> {
        self.calls.push("finish");
        Ok(())
    }
    fn close(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        0
    }
}
fn term(freq: i32) -> TermVectorTerm {
    TermVectorTerm {
        term: vec![],
        freq,
        positions: None,
        start_offsets: None,
        end_offsets: None,
        payloads: None,
    }
}
fn field(term: TermVectorTerm, p: bool, o: bool) -> VectorDocument {
    vec![(
        "f".into(),
        TermVectorField {
            field_number: 9,
            has_positions: p,
            has_offsets: o,
            has_payloads: false,
            terms: vec![term],
        },
    )]
}
#[test]
fn superclass_no_positions_and_invalid_occurrences() {
    let infos = FieldInfos::new(vec![FieldInfo::new("f", 7)]).unwrap();
    let mut p = Probe::default();
    add_document(&mut p, Some(field(term(3), false, false)), &infos).unwrap();
    assert!(p.positions.is_empty());
    assert!(p.calls.contains(&"endterm"));
    let mut t = term(1);
    t.start_offsets = Some(vec![2]);
    t.end_offsets = Some(vec![5]);
    add_document(&mut p, Some(field(t, false, true)), &infos).unwrap();
    assert_eq!(p.positions, vec![(-1, 2, 5, None)]);
    assert_eq!(
        add_document(&mut p, Some(field(term(1), true, false)), &infos)
            .unwrap_err()
            .to_string(),
        "missing term-vector occurrence"
    );
    assert!(add_document(&mut p, Some(field(term(-1), true, false)), &infos).is_err());
    let empty = FieldInfos::new(vec![]).unwrap();
    assert_eq!(
        add_document(&mut p, Some(field(term(0), false, false)), &empty)
            .unwrap_err()
            .to_string(),
        "missing destination field"
    );
}
#[test]
fn proximity_wrapping_truncation_and_payload_distinction() {
    let mut p = Probe::default();
    // Java's unsigned position delta, then wrapping int addition.
    p.add_prox(
        3,
        Some(&mut SliceInput::new(&[
            0xfe, 0xff, 0xff, 0xff, 0x0f, 2, 1, 0,
        ])),
        None,
    )
    .unwrap();
    assert_eq!(
        p.positions,
        vec![
            (i32::MAX, -1, -1, None),
            (i32::MIN, -1, -1, None),
            (i32::MIN, -1, -1, Some(vec![]))
        ]
    );
    let mut large = vec![1, 33];
    large.extend(0_u8..33);
    p.add_prox(1, Some(&mut SliceInput::new(&large)), None)
        .unwrap();
    assert_eq!(
        p.positions.last().unwrap(),
        &(0, -1, -1, Some((0_u8..33).collect()))
    );
    let mut input = SliceInput::new(&[1, 2, 9]);
    assert_eq!(
        p.add_prox(1, Some(&mut input), None).unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert_eq!(input.position(), 2);
    let mut input = SliceInput::new(&[1, 0xff, 0xff, 0xff, 0xff, 0x0f]);
    assert!(p.add_prox(1, Some(&mut input), None).is_err());
    let mut input = SliceInput::new(&[0x80, 0x80, 0x80, 0x80, 0x80, 0]);
    assert!(p.add_prox(1, Some(&mut input), None).is_err());
    p.add_prox(1, None, None).unwrap();
    assert_eq!(p.positions.last().unwrap(), &(-1, -1, -1, None));
    assert_eq!(
        input_error(lucene_store::Error::Io(io::Error::other("typed"))).to_string(),
        "typed"
    );
}
#[test]
fn sorted_merge_keeps_current_source_below_queue_minimum() {
    let infos = FieldInfos::new(vec![]).unwrap();
    let sources = [
        MergeSource {
            reader: None,
            doc_map: &[0, 1],
        },
        MergeSource {
            reader: None,
            doc_map: &[2, 3],
        },
    ];
    let mut abort = || Ok(());
    let mut state = MergeState {
        sources: &sources,
        field_infos: &infos,
        needs_index_sort: true,
        check_aborted: &mut abort,
    };
    let mut writer = Probe::default();
    assert_eq!(writer.merge(&mut state).unwrap(), 4);
    assert_eq!(
        writer.calls,
        vec!["doc", "enddoc", "doc", "enddoc", "doc", "enddoc", "doc", "enddoc", "finish"]
    );
}
#[test]
fn empty_merge_and_simpletext_lifecycle() {
    let infos = FieldInfos::new(vec![]).unwrap();
    let sources = [MergeSource {
        reader: None,
        doc_map: &[-1, -1],
    }];
    let mut abort = || Ok(());
    let mut p = Probe::default();
    let mut state = MergeState {
        sources: &sources,
        field_infos: &infos,
        needs_index_sort: true,
        check_aborted: &mut abort,
    };
    assert_eq!(p.merge(&mut state).unwrap(), 0);
    assert_eq!(p.calls, vec!["finish"]);
    let dir = ByteBuffersDirectory::new();
    let state = VectorState {
        directory: &dir,
        segment: "_x",
        max_doc: 1,
    };
    let mut format = SimpleTextVectorsFormat;
    assert!(format.vectors_reader(&state).is_err());
    let mut writer = format.vectors_writer(&state).unwrap();
    assert_eq!(writer.ram_bytes_used(), 0);
    writer.start_document(0).unwrap();
    assert!(writer.finish(2).is_err());
    writer.close().unwrap();
    writer.close().unwrap();
    assert!(writer.start_document(0).is_err());
    dir.delete_file("_x.tvc").unwrap();
    let mut writer = format.vectors_writer(&state).unwrap();
    writer.close().unwrap();
    assert!(writer.finish(0).is_err());
    assert!(SimpleTextVectorsReader::open(vec![], 0).is_err());
}
