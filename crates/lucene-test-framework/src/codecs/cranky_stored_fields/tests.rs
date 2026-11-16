#![cfg(test)]
use super::*;
use lucene_store::data_input::SliceInput;
use lucene_store::index_output::{CloseVerdict, OutputFaults, WriteVerdict};
use lucene_store::ByteBuffersDirectory;
const ID: [u8; 16] = [0; 16];
fn state(dir: &dyn Directory) -> StoredFieldsState<'_> {
    StoredFieldsState {
        directory: dir,
        segment: "_s",
        id: &ID,
        suffix: "x",
        mode: Mode::BestSpeed,
    }
}
struct FailingWriter {
    random: Rc<RefCell<JavaRandom>>,
    calls: usize,
}
impl StoredFieldsWriter for FailingWriter {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.calls += 1;
        assert!(self.random.try_borrow_mut().is_ok());
        Err(error("delegate failure"))
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.start_document()
    }
    fn write_field(&mut self, _: i32, _: &FieldValue) -> stored_fields::Result<()> {
        self.start_document()
    }
    fn write_field_data_input(
        &mut self,
        _: i32,
        _: &mut dyn DataInput,
        _: usize,
    ) -> stored_fields::Result<()> {
        self.start_document()
    }
    fn finish(&mut self, _: i32) -> stored_fields::Result<()> {
        self.start_document()
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.start_document()
    }
    fn ram_bytes_used(&self) -> usize {
        91
    }
    fn child_resources(&self) -> &[ChildResource] {
        &[]
    }
}
fn first(bound: i32) -> i64 {
    (0..100000)
        .find(|&seed| JavaRandom::new(seed).next_int_bounded(bound) == 0)
        .unwrap()
}
#[test]
fn field_guards_do_not_consume_input_and_delegate_failures_release_rng() {
    let random = Rc::new(RefCell::new(JavaRandom::new(first(10000))));
    let mut writer = CrankyStoredFieldsWriter::new(
        FailingWriter {
            random: random.clone(),
            calls: 0,
        },
        random.clone(),
    );
    let mut input = SliceInput::new(&[1, 2]);
    assert_eq!(
        writer
            .write_field_data_input(0, &mut input, 2)
            .unwrap_err()
            .to_string(),
        FAKE_FIELD_ERROR
    );
    assert_eq!(input.position(), 0);
    assert_eq!(writer.delegate().calls, 0);
    for (bound, op) in [(10000, 0), (10000, 1), (10000, 2), (100, 3), (100, 4)] {
        random.borrow_mut().set_seed(first(bound));
        let err = match op {
            0 => writer.start_document(),
            1 => writer.finish_document(),
            2 => writer.write_field(0, &FieldValue::Int(2)),
            3 => writer.finish(0),
            _ => writer
                .merge(&MergeState {
                    sources: &[],
                    needs_index_sort: false,
                    check_aborted: &|| Ok(()),
                })
                .map(|_| ()),
        }
        .unwrap_err();
        assert!(err.to_string().starts_with("Fake IOException"));
    }
    assert_eq!(writer.delegate().calls, 0);
    random.borrow_mut().set_seed(42);
    assert_eq!(
        writer
            .write_field_data_input(0, &mut input, 2)
            .unwrap_err()
            .to_string(),
        "delegate failure"
    );
    assert_eq!(input.position(), 0);
    let before = random.borrow().clone().next_int();
    assert_eq!(writer.close().unwrap_err().to_string(), "delegate failure");
    assert_eq!(random.borrow().clone().next_int(), before);
    writer.delegate_mut().calls = 13;
    assert_eq!(writer.delegate().calls, 13);
    assert_eq!(writer.ram_bytes_used(), 91);
    assert!(writer.child_resources().is_empty());
}
struct FailingClose;
impl OutputFaults for FailingClose {
    fn before_write(&mut self, _: &str, _: usize) -> WriteVerdict {
        WriteVerdict::Write
    }
    fn on_close(&mut self, _: &str) -> CloseVerdict {
        CloseVerdict::FailAfterFlush(std::io::Error::other("close failed"))
    }
}
#[test]
fn concrete_validation_creation_finish_and_close_boundaries() {
    let mut format = Lucene90StoredFieldsFormat;
    let dir = ByteBuffersDirectory::new();
    dir.create_output("_s_x.fdm").unwrap().close().unwrap();
    assert!(format.fields_writer(state(&dir)).is_err());
    assert!(dir.open("_s_x.fdt").is_err());
    let dir = ByteBuffersDirectory::new();
    dir.create_output("_s_x.fdt").unwrap().close().unwrap();
    assert!(format.fields_writer(state(&dir)).is_err());
    assert!(!dir.open("_s_x.fdm").unwrap().is_empty());
    let dir = ByteBuffersDirectory::new();
    let mut writer = format.fields_writer(state(&dir)).unwrap();
    assert!(writer
        .finish(1)
        .unwrap_err()
        .to_string()
        .contains("Wrote 0 docs"));
    assert!(writer
        .write_field_data_input(0, &mut SliceInput::new(&[]), 1)
        .is_err());
    writer.start_document().unwrap();
    writer.finish_document().unwrap();
    dir.create_output("_s_x.fdx").unwrap().close().unwrap();
    assert!(writer.finish(1).is_err());
    writer.close().unwrap();
    assert!(writer.finish_document().is_err());
    assert!(writer.write_field(0, &FieldValue::Int(1)).is_err());
    assert!(writer.finish(1).is_err());
    let dir = ByteBuffersDirectory::new();
    let mut writer = format.fields_writer(state(&dir)).unwrap();
    writer
        .outputs
        .as_mut()
        .unwrap()
        .0
        .set_faults(Box::new(FailingClose));
    assert_eq!(writer.close().unwrap_err().to_string(), "close failed");
    assert!(!dir.open("_s_x.fdt").unwrap().is_empty());
    writer.close().unwrap();
}
#[test]
fn empty_merge_and_abort_before_sources_leave_writer_unmodified() {
    let dir = ByteBuffersDirectory::new();
    let random = Rc::new(RefCell::new(JavaRandom::new(first(100))));
    let mut format = CrankyStoredFieldsFormat::new(Lucene90StoredFieldsFormat, random.clone());
    assert_eq!(
        format.fields_writer(state(&dir)).err().unwrap().to_string(),
        FAKE_OPEN_ERROR
    );
    assert!(dir.list_all().unwrap().is_empty());
    let mut writer = format.delegate_mut().fields_writer(state(&dir)).unwrap();
    assert_eq!(
        writer
            .merge(&MergeState {
                sources: &[],
                needs_index_sort: false,
                check_aborted: &|| Ok(())
            })
            .unwrap(),
        0
    );
    writer.close().unwrap();
    assert_eq!(std::mem::size_of_val(format.delegate()), 0);
    assert!(format
        .delegate_mut()
        .fields_reader(StoredFieldsReadState {
            fdt: &[],
            fdx: &[],
            fdm: &[],
            id: &ID,
            suffix: ""
        })
        .is_err());
}
struct UnreadableSource;
impl StoredFieldsSource for UnreadableSource {
    fn check_integrity(&self) -> stored_fields::Result<()> {
        panic!("abort must precede integrity")
    }
    fn visit_document(&self, _: i32, _: &mut dyn StoredFieldVisitor) -> stored_fields::Result<()> {
        panic!("aborted source visited")
    }
}
#[test]
fn abort_precedes_source_callbacks_and_close_failure_follows_delegation() {
    let dir = ByteBuffersDirectory::new();
    let mut format = Lucene90StoredFieldsFormat;
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut writer =
        CrankyStoredFieldsWriter::new(format.fields_writer(state(&dir)).unwrap(), random.clone());
    let sources = [MergeSource {
        reader: &UnreadableSource,
        doc_map: &[0],
        field_map: &[],
    }];
    assert_eq!(
        writer
            .merge(&MergeState {
                sources: &sources,
                needs_index_sort: false,
                check_aborted: &|| Err(error("aborted"))
            })
            .unwrap_err()
            .to_string(),
        "aborted"
    );
    assert_eq!(writer.delegate().completed, 0);
    let mut expected = JavaRandom::new(42);
    expected.next_int_bounded(100);
    assert_eq!(random.borrow().clone().next_int(), expected.next_int());
    random.borrow_mut().set_seed(first(1000));
    assert_eq!(writer.close().unwrap_err().to_string(), FAKE_CLOSE_ERROR);
    assert!(writer.delegate().outputs.is_none());
    assert!(!dir.open("_s_x.fdt").unwrap().is_empty());
    assert!(writer
        .write_field_data_input(0, &mut SliceInput::new(&[]), 0)
        .is_err());
}
#[test]
fn every_delegate_callback_propagates_failure_after_exactly_one_draw() {
    for op in 0..4 {
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut writer = CrankyStoredFieldsWriter::new(
            FailingWriter {
                random: random.clone(),
                calls: 0,
            },
            random.clone(),
        );
        let result = match op {
            0 => writer.start_document(),
            1 => writer.finish_document(),
            2 => writer.write_field(7, &FieldValue::Int(3)),
            _ => writer.finish(1),
        };
        assert_eq!(result.unwrap_err().to_string(), "delegate failure");
        assert_eq!(writer.delegate().calls, 1);
        let mut expected = JavaRandom::new(42);
        expected.next_int_bounded(if op == 3 { 100 } else { 10000 });
        assert_eq!(random.borrow_mut().next_int(), expected.next_int());
    }
}
