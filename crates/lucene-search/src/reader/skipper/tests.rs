#![cfg(test)]
use super::*;
use lucene_codecs::doc_values::{SkipIndexInterval, SkipIndexLevelInterval};
use lucene_codecs::field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo};
use lucene_codecs::simpletext::doc_values_reader::SimpleTextDocValuesReader;
use lucene_codecs::simpletext::doc_values_writer::SimpleTextDocValuesWriter;

fn check(mut skipper: DocValuesSkipper<'_>) {
    assert_eq!(skipper.num_levels(), 1);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (-1, -1));
    assert_eq!(
        (skipper.global_min_value(), skipper.global_max_value()),
        (0, 15)
    );
    assert_eq!(
        (skipper.global_doc_count(), skipper.max_value_count()),
        (16, 1)
    );
    skipper.advance(0);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (0, 7));
    assert_eq!(
        (
            skipper.min_value(0),
            skipper.max_value(0),
            skipper.doc_count(0)
        ),
        (0, 7, 8)
    );
    skipper.advance_range(9, 10);
    assert_eq!((skipper.min_doc_id(0), skipper.max_doc_id(0)), (8, 15));
    assert_eq!(
        (
            skipper.min_value(0),
            skipper.max_value(0),
            skipper.doc_count(0)
        ),
        (8, 15, 8)
    );
    skipper.advance_range(100, 100);
    assert_eq!(skipper.min_doc_id(0), NO_MORE_DOCS);
    assert_eq!(skipper.max_doc_id(0), NO_MORE_DOCS);
}

#[test]
fn value_range_skips_a_whole_nonintersecting_upper_level() {
    let index = DocValuesSkipIndex {
        min_value: 0,
        max_value: 15,
        doc_count: 16,
        max_doc_id: 15,
        max_value_count: 1,
        intervals: vec![SkipIndexInterval {
            levels: vec![
                SkipIndexLevelInterval {
                    min_doc_id: 0,
                    max_doc_id: 7,
                    min_value: 0,
                    max_value: 7,
                    doc_count: 8,
                },
                SkipIndexLevelInterval {
                    min_doc_id: 0,
                    max_doc_id: 15,
                    min_value: 0,
                    max_value: 15,
                    doc_count: 16,
                },
            ],
        }],
    };
    let mut skipper = DocValuesSkipper::new(&index);
    skipper.advance(0);
    assert_eq!(skipper.num_levels(), 2);
    skipper.advance_range(100, 100);
    assert_eq!(skipper.min_doc_id(0), NO_MORE_DOCS);
}

#[test]
fn binary_and_simpletext_cursors_keep_their_codec_intervals() {
    let index = DocValuesSkipIndex {
        min_value: 0,
        max_value: 15,
        doc_count: 16,
        max_doc_id: 15,
        max_value_count: 1,
        intervals: [0i32, 8]
            .into_iter()
            .map(|min| SkipIndexInterval {
                levels: vec![SkipIndexLevelInterval {
                    min_doc_id: min,
                    max_doc_id: min.saturating_add(7),
                    min_value: i64::from(min),
                    max_value: i64::from(min.saturating_add(7)),
                    doc_count: 8,
                }],
            })
            .collect(),
    };
    check(DocValuesSkipper::new(&index));
    let mut field = FieldInfo::new("n", 0);
    field.doc_values_type = DocValuesType::Numeric;
    field.doc_values_skip_index_type = DocValuesSkipIndexType::Range;
    let mut writer = SimpleTextDocValuesWriter::new(16);
    writer
        .add_numeric_field(
            &field,
            &(0..16).map(|doc| (doc, i64::from(doc))).collect::<Vec<_>>(),
        )
        .unwrap();
    let data = writer.finish().unwrap();
    let reader = SimpleTextDocValuesReader::open(&data, 16).unwrap();
    check(DocValuesSkipper::simpletext(
        reader.skipper("n", DocValuesType::Numeric).unwrap(),
    ));
    let mut initial_range =
        DocValuesSkipper::simpletext(reader.skipper("n", DocValuesType::Numeric).unwrap());
    initial_range.advance_range(9, 10);
    assert_eq!(initial_range.min_doc_id(0), 8);
}

/// One fixed interval, at `-1` until advanced.
#[derive(Debug, Clone)]
struct Fixed(i32);

impl SkipperImpl for Fixed {
    fn advance(&mut self, target: i32) {
        self.0 = if target > 9 { NO_MORE_DOCS } else { 0 };
    }
    fn num_levels(&self) -> usize {
        1
    }
    fn min_doc_id(&self, _level: usize) -> i32 {
        self.0
    }
    fn max_doc_id(&self, _level: usize) -> i32 {
        if self.0 == 0 {
            9
        } else {
            self.0
        }
    }
    fn min_value(&self, _level: usize) -> i64 {
        3
    }
    fn max_value(&self, _level: usize) -> i64 {
        4
    }
    fn doc_count(&self, _level: usize) -> i32 {
        10
    }
    fn global_min_value(&self) -> i64 {
        3
    }
    fn global_max_value(&self) -> i64 {
        4
    }
    fn global_doc_count(&self) -> i32 {
        10
    }
    fn max_value_count(&self) -> i32 {
        1
    }
    fn clone_box<'s>(&self) -> Box<dyn SkipperImpl + 's>
    where
        Self: 's,
    {
        Box::new(self.clone())
    }
}

#[test]
fn a_skipper_outside_the_codecs_dispatches_through_its_impl() {
    let mut s = DocValuesSkipper::from_impl(Box::new(Fixed(-1)));
    assert_eq!(
        (s.min_doc_id(0), s.max_doc_id(0), s.num_levels()),
        (-1, -1, 1)
    );
    assert_eq!(
        (
            s.global_min_value(),
            s.global_max_value(),
            s.global_doc_count()
        ),
        (3, 4, 10)
    );
    assert_eq!(s.max_value_count(), 1);
    let copy = s.clone();
    s.advance_range(3, 3);
    assert_eq!((s.min_doc_id(0), s.max_doc_id(0)), (0, 9));
    assert_eq!((s.min_value(0), s.max_value(0), s.doc_count(0)), (3, 4, 10));
    assert_eq!(copy.min_doc_id(0), -1, "the copy kept its position");
    s.advance_range(5, 6);
    assert_eq!(s.min_doc_id(0), NO_MORE_DOCS);
    assert!(format!("{copy:?}").contains("Fixed"));
}
