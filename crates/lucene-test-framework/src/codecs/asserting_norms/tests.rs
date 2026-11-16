#![cfg(test)]
use super::*;
use lucene_store::ByteBuffersDirectory;
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn field(number: i32) -> FieldInfo {
    let mut field = FieldInfo::new(format!("norm{number}"), number);
    field.index_options = IndexOptions::Docs;
    field
}
fn error() -> lucene_search::Error {
    lucene_store::Error::Io(std::io::Error::other("source failure")).into()
}
struct Source {
    gets: Cell<usize>,
    fail_get: usize,
    fail_next: bool,
    docs: Vec<(i32, i64)>,
}
impl Source {
    fn sparse() -> Self {
        Self {
            gets: Cell::new(0),
            fail_get: 0,
            fail_next: false,
            docs: vec![(0, 7), (2, 9)],
        }
    }
}
struct Cursor<'a> {
    source: &'a Source,
    position: usize,
    doc: i32,
}
impl DocIdSetIterator for Cursor<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.source.fail_next {
            return Err(error());
        }
        self.doc = self
            .source
            .docs
            .get(self.position)
            .map_or(NO_MORE_DOCS, |pair| pair.0);
        self.position += 1;
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        while self.next_doc()? < target {}
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.source.docs.len() as i64
    }
}
impl DocValuesIterator for Cursor<'_> {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        Ok(self.advance(target)? == target)
    }
}
impl NumericDocValues for Cursor<'_> {
    fn long_value(&self) -> i64 {
        self.source.docs[self.position - 1].1
    }
}
impl NormsProducer for Source {
    fn get_norms(&self, _: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        let gets = self.gets.get() + 1;
        self.gets.set(gets);
        if gets == self.fail_get {
            Err(error())
        } else {
            Ok(Box::new(Cursor {
                source: self,
                position: 0,
                doc: -1,
            }))
        }
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
    fn check_integrity(&self) -> Result<()> {
        Ok(())
    }
    fn get_merge_instance(&self) -> Self {
        Self::sparse()
    }
}
#[derive(Default)]
struct Target {
    adds: usize,
    closes: usize,
}
impl NormsConsumer for Target {
    fn add_norms_field(&mut self, _: &FieldInfo, _: &dyn NormsProducer) -> Result<()> {
        self.adds += 1;
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.closes += 1;
        Ok(())
    }
}

#[test]
fn exposed_delegates_control_and_observe_the_same_wrapped_objects() {
    let mut source = AssertingNormsProducer::new(Source::sparse(), 3);
    source.delegate_mut().fail_get = 1;
    assert_eq!(
        source.get_norms(&field(0)).err().unwrap().to_string(),
        "source failure"
    );
    assert_eq!(source.delegate().gets.get(), 1);
    let mut target = AssertingNormsConsumer::new(Target::default(), 3);
    target.delegate_mut().closes = 4;
    target
        .add_norms_field(&field(0), &source.delegate().get_merge_instance())
        .unwrap();
    target.close().unwrap();
    assert_eq!((target.delegate().adds, target.delegate().closes), (1, 6));
    let mut unindexed = field(0);
    unindexed.index_options = IndexOptions::None;
    assert!(catch_unwind(AssertUnwindSafe(|| source.get_norms(&unindexed))).is_err());
    assert_eq!(
        source.delegate().gets.get(),
        1,
        "field check precedes delegate"
    );
}

#[test]
fn native_writer_replays_all_passes_and_propagates_each_source_failure() {
    for fail_get in 1..=3 {
        let directory = ByteBuffersDirectory::new();
        let state = NormsState {
            directory: &directory,
            segment: "_n",
            id: &[0; 16],
            suffix: "",
            max_doc: 3,
        };
        let mut writer = Lucene90NormsFormat.norms_consumer(&state).unwrap();
        let source = Source {
            fail_get,
            ..Source::sparse()
        };
        assert_eq!(
            writer
                .add_norms_field(&field(0), &source)
                .unwrap_err()
                .to_string(),
            "source failure"
        );
        assert_eq!(source.gets.get(), fail_get);
        writer.close().unwrap();
    }
    let directory = ByteBuffersDirectory::new();
    let state = NormsState {
        directory: &directory,
        segment: "_n",
        id: &[0; 16],
        suffix: "",
        max_doc: 3,
    };
    let mut writer = Lucene90NormsFormat.norms_consumer(&state).unwrap();
    let source = Source {
        fail_next: true,
        ..Source::sparse()
    };
    assert_eq!(
        writer
            .add_norms_field(&field(0), &source)
            .unwrap_err()
            .to_string(),
        "source failure"
    );
    let source = Source::sparse();
    writer.add_norms_field(&field(0), &source).unwrap();
    assert_eq!(source.gets.get(), 3);
    writer.close().unwrap();
    assert!(writer.add_norms_field(&field(1), &source).is_err());
    let mut producer = Lucene90NormsFormat
        .norms_producer(
            &state,
            &FieldInfos {
                fields: vec![field(0)],
            },
        )
        .unwrap();
    assert!(matches!(
        producer.get_norms(&field(9)).err(),
        Some(lucene_search::Error::Norms(
            norms::Error::UnknownFieldNumber(9)
        ))
    ));
    let mut cursor = producer.get_norms(&field(0)).unwrap();
    assert_eq!(cursor.advance(1).unwrap(), 2);
    assert_eq!(cursor.long_value(), 9);
    assert!(!cursor.advance_exact(1).unwrap());
    assert_eq!(cursor.doc_id(), 1);
    assert_eq!(cursor.next_doc().unwrap(), 2);
    assert_eq!(cursor.long_value(), 9);
    assert!(cursor.advance_exact(2).unwrap());
    assert_eq!(cursor.long_value(), 9);
    assert!(cursor.advance_exact(0).unwrap());
    assert_eq!(cursor.long_value(), 7);
    assert!(cursor.advance_exact(-1).is_err());
    assert!(cursor.advance_exact(3).is_err());
    assert_eq!(cursor.advance(3).unwrap(), NO_MORE_DOCS);
    drop(cursor);
    let merging = producer.get_merge_instance();
    producer.close().unwrap();
    producer.close().unwrap();
    assert_eq!(producer.to_string(), "Lucene90NormsProducer(fields=1)");
    let mut independent = merging.get_norms(&field(0)).unwrap();
    assert_eq!(independent.next_doc().unwrap(), 0);
    assert_eq!(independent.long_value(), 7);
    merging.check_integrity().unwrap();
    let clone = producer.get_merge_instance();
    assert_eq!(
        clone.get_norms(&field(0)).err().unwrap().to_string(),
        "closed norms producer"
    );
    assert_eq!(
        producer.check_integrity().unwrap_err().to_string(),
        "closed norms producer"
    );
}

#[test]
fn native_open_and_close_errors_are_returned() {
    let directory = ByteBuffersDirectory::new();
    let state = NormsState {
        directory: &directory,
        segment: "_n",
        id: &[0; 16],
        suffix: "",
        max_doc: 3,
    };
    assert!(Lucene90NormsFormat
        .norms_producer(&state, &FieldInfos { fields: vec![] })
        .is_err());
    let mut writer = Lucene90NormsFormat.norms_consumer(&state).unwrap();
    let invalid = Source {
        docs: vec![(3, 1)],
        ..Source::sparse()
    };
    writer.add_norms_field(&field(0), &invalid).unwrap();
    assert!(writer.close().is_err());
    writer.close().unwrap();
}

#[test]
fn metadata_field_errors_precede_opening_missing_data() {
    use lucene_store::{DataOutput, Directory};
    let directory = ByteBuffersDirectory::new();
    let (metadata, _) =
        norms::write_fields(&[NormsField::Dense(0, &[1, 2, 3])], 3, &[0; 16], "").unwrap();
    let mut output = directory.create_output("_n.nvm").unwrap();
    output.write_bytes(&metadata);
    output.close().unwrap();
    let state = NormsState {
        directory: &directory,
        segment: "_n",
        id: &[0; 16],
        suffix: "",
        max_doc: 3,
    };
    let unknown = Lucene90NormsFormat
        .norms_producer(&state, &FieldInfos { fields: vec![] })
        .err()
        .unwrap();
    assert!(matches!(
        unknown,
        lucene_search::Error::Norms(norms::Error::UnknownFieldNumber(0))
    ));
    let mut omitted = field(0);
    omitted.omit_norms = true;
    let invalid = Lucene90NormsFormat
        .norms_producer(
            &state,
            &FieldInfos {
                fields: vec![omitted],
            },
        )
        .err()
        .unwrap();
    assert!(matches!(
        invalid,
        lucene_search::Error::Norms(norms::Error::FieldHasNoNorms(_))
    ));
    assert!(Lucene90NormsFormat
        .norms_producer(
            &state,
            &FieldInfos {
                fields: vec![field(0)]
            }
        )
        .is_err());
}

#[test]
fn incomplete_native_data_returns_an_integrity_error() {
    let incomplete = NativeNorms {
        metadata: norms::Norms { entries: vec![] },
        data: vec![],
    };
    assert_eq!(
        incomplete.check_integrity().unwrap_err().to_string(),
        "missing norms data footer"
    );
}
