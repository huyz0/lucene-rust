//! `DocValuesSkipper` at the reader boundary, dispatching to each codec's
//! cursor. Java's abstract class is a Rust enum over borrowed codec cursors;
//! interval traversal remains in the codec. `advance_range` is Java's
//! format-independent default method. A skipper that is neither (Java's
//! other subclasses: the test framework's `AssertingDocValuesSkipper`)
//! implements [`SkipperImpl`] and is wrapped with
//! [`DocValuesSkipper::from_impl`].

use lucene_codecs::doc_values::{DocValuesSkipIndex, DocValuesSkipper as BinarySkipper};
use lucene_codecs::simpletext::doc_values_reader::SimpleTextSkipper;

use super::NO_MORE_DOCS;

/// `index.DocValuesSkipper`, independent of the segment's codec.
#[derive(Debug, Clone)]
pub struct DocValuesSkipper<'a>(Kind<'a>);

#[derive(Debug)]
enum Kind<'a> {
    Binary(BinarySkipper<'a>),
    SimpleText(SimpleTextSkipper<'a>),
    Other(Box<dyn SkipperImpl + 'a>),
}

impl Clone for Kind<'_> {
    fn clone(&self) -> Self {
        match self {
            Kind::Binary(cursor) => Kind::Binary(cursor.clone()),
            Kind::SimpleText(cursor) => Kind::SimpleText(cursor.clone()),
            Kind::Other(other) => Kind::Other(other.clone_box()),
        }
    }
}

/// `DocValuesSkipper`'s abstract methods, for a skipper outside the codecs
/// (a wrapper such as the test framework's asserting one). Levels are
/// `usize`, as [`DocValuesSkipper`]'s getters take them.
pub trait SkipperImpl: std::fmt::Debug + Send {
    /// `advance(target)`.
    fn advance(&mut self, target: i32);
    /// `numLevels()`.
    fn num_levels(&self) -> usize;
    /// `minDocID(level)`.
    fn min_doc_id(&self, level: usize) -> i32;
    /// `maxDocID(level)`.
    fn max_doc_id(&self, level: usize) -> i32;
    /// `minValue(level)`.
    fn min_value(&self, level: usize) -> i64;
    /// `maxValue(level)`.
    fn max_value(&self, level: usize) -> i64;
    /// `docCount(level)`.
    fn doc_count(&self, level: usize) -> i32;
    /// `minValue()`.
    fn global_min_value(&self) -> i64;
    /// `maxValue()`.
    fn global_max_value(&self) -> i64;
    /// `docCount()`.
    fn global_doc_count(&self) -> i32;
    /// `maxValueCount()`.
    fn max_value_count(&self) -> i32;
    /// A copy at the same position (the skipper is `Clone`).
    fn clone_box<'s>(&self) -> Box<dyn SkipperImpl + 's>
    where
        Self: 's;
}

macro_rules! level_getter {
    ($method:ident, $text_method:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $method(&self, level: usize) -> $ty {
            match &self.0 {
                Kind::Binary(cursor) => cursor.$method(level),
                Kind::SimpleText(cursor) => cursor.$text_method(level as i32),
                Kind::Other(other) => other.$method(level),
            }
        }
    };
}
macro_rules! global_getter {
    ($method:ident, $text_method:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $method(&self) -> $ty {
            match &self.0 {
                Kind::Binary(cursor) => cursor.$method(),
                Kind::SimpleText(cursor) => cursor.$text_method(),
                Kind::Other(other) => other.$method(),
            }
        }
    };
}

impl<'a> DocValuesSkipper<'a> {
    /// A skipper over the default codec's parsed skip index.
    pub fn new(index: &'a DocValuesSkipIndex) -> Self {
        Self(Kind::Binary(BinarySkipper::new(index)))
    }

    pub(super) fn simpletext(cursor: SimpleTextSkipper<'a>) -> Self {
        Self(Kind::SimpleText(cursor))
    }

    /// A skipper implemented outside the codecs (a `DocValuesSkipper`
    /// subclass in Java).
    pub fn from_impl(skipper: Box<dyn SkipperImpl + 'a>) -> Self {
        Self(Kind::Other(skipper))
    }

    /// `advance(target)`: the first interval that may contain a document
    /// at or after `target`; target must exceed the current level-0 maximum.
    pub fn advance(&mut self, target: i32) {
        match &mut self.0 {
            Kind::Binary(cursor) => cursor.advance(target),
            Kind::SimpleText(cursor) => cursor.advance(target),
            Kind::Other(other) => other.advance(target),
        }
    }

    /// `numLevels()` at the current position.
    pub fn num_levels(&self) -> usize {
        match &self.0 {
            Kind::Binary(cursor) => cursor.num_levels(),
            Kind::SimpleText(cursor) => cursor.num_levels() as usize,
            Kind::Other(other) => other.num_levels(),
        }
    }

    level_getter!(
        min_doc_id,
        min_doc_id,
        i32,
        "Minimum document ID at this level; -1 before advance, NO_MORE_DOCS after exhaustion."
    );
    level_getter!(
        max_doc_id,
        max_doc_id,
        i32,
        "Maximum document ID at this level; -1 before advance, NO_MORE_DOCS after exhaustion."
    );
    level_getter!(
        min_value,
        min_value_at,
        i64,
        "Minimum value bound at this level."
    );
    level_getter!(
        max_value,
        max_value_at,
        i64,
        "Maximum value bound at this level."
    );
    level_getter!(
        doc_count,
        doc_count_at,
        i32,
        "Number of documents with values in this interval."
    );
    global_getter!(
        global_min_value,
        min_value,
        i64,
        "Field-wide minimum value bound."
    );
    global_getter!(
        global_max_value,
        max_value,
        i64,
        "Field-wide maximum value bound."
    );
    global_getter!(
        global_doc_count,
        doc_count,
        i32,
        "Field-wide number of documents with values."
    );
    global_getter!(
        max_value_count,
        max_value_count,
        i32,
        "Maximum values per document, or -1 if unknown."
    );

    /// Java: `DocValuesSkipper.advance(long, long)`. Advance past intervals
    /// whose value bounds do not intersect the inclusive query range.
    pub fn advance_range(&mut self, min: i64, max: i64) {
        if self.min_doc_id(0) == -1 {
            self.advance(0);
        }
        while self.min_doc_id(0) != NO_MORE_DOCS
            && (self.min_value(0) > max || self.max_value(0) < min)
        {
            let mut max_doc = self.max_doc_id(0);
            let mut level = 1usize;
            while level < self.num_levels()
                && (self.min_value(level) > max || self.max_value(level) < min)
            {
                max_doc = self.max_doc_id(level);
                level = level.saturating_add(1);
            }
            self.advance(max_doc.saturating_add(1));
        }
    }
}

#[cfg(test)]
mod tests {
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
}
