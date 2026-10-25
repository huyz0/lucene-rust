//! `DocValuesSkipper` at the reader boundary, dispatching to each codec's
//! cursor. Java's abstract class is a Rust enum over borrowed codec cursors;
//! interval traversal remains in the codec. `advance_range` is Java's
//! format-independent default method.

use lucene_codecs::doc_values::{DocValuesSkipIndex, DocValuesSkipper as BinarySkipper};
use lucene_codecs::simpletext::doc_values_reader::SimpleTextSkipper;

use super::NO_MORE_DOCS;

/// `index.DocValuesSkipper`, independent of the segment's codec.
#[derive(Debug, Clone)]
pub struct DocValuesSkipper<'a>(Kind<'a>);

#[derive(Debug, Clone)]
enum Kind<'a> {
    Binary(BinarySkipper<'a>),
    SimpleText(SimpleTextSkipper<'a>),
}

macro_rules! level_getter {
    ($method:ident, $text_method:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $method(&self, level: usize) -> $ty {
            match &self.0 {
                Kind::Binary(cursor) => cursor.$method(level),
                Kind::SimpleText(cursor) => cursor.$text_method(level as i32),
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

    /// `advance(target)`: the first interval that may contain a document
    /// at or after `target`; target must exceed the current level-0 maximum.
    pub fn advance(&mut self, target: i32) {
        match &mut self.0 {
            Kind::Binary(cursor) => cursor.advance(target),
            Kind::SimpleText(cursor) => cursor.advance(target),
        }
    }

    /// `numLevels()` at the current position.
    pub fn num_levels(&self) -> usize {
        match &self.0 {
            Kind::Binary(cursor) => cursor.num_levels(),
            Kind::SimpleText(cursor) => cursor.num_levels() as usize,
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
}
