//! Java `RandomCodec` installed in
//! [`lucene_index::index_writer::IndexWriter`] (`IndexWriter::set_codec`):
//! an `AssertingCodec` (the segment records `Asserting`) whose per-field
//! postings and doc-values formats are [`RandomCodec`]'s shuffled
//! candidates, chosen by `Math.abs(perFieldSeed ^ field.hashCode()) % size`,
//! and whose points are written by `RandomlySplittingBKDWriter`.
//!
//! Each candidate becomes its native format: `Lucene104(min, max)`,
//! `FST50`, `Direct` (its `Lucene104` delegate's files under its own name),
//! `TestBloomFilteredLucenePostings`, `MockRandom` (seeded as Java's),
//! `BlockTreeOrds(min, max)`, the three block-terms formats with their
//! intervals, default `Lucene104` and `Asserting`; doc values are the two
//! default `Lucene90` instances and `Asserting`. Every candidate is its own
//! instance, as in Java, so equal settings still get separate suffixes
//! (`_Lucene90_0`, `_Lucene90_1`). Points go through the asserting writer
//! to a `Lucene90PointsWriter` with the codec's leaf size and heap budget
//! whose every field, flushed or merged, is added to a fresh `BKDWriter`
//! that splits each cell on `new Random(bkdSplitRandomSeed ^
//! field.hashCode()).nextInt(numIndexDims)`.
//!
//! Differs: a field whose selection is `Math.abs(Integer.MIN_VALUE)`, or a
//! codec whose every candidate was avoided, panics with Java's exception
//! message where Java throws from the writer, as does the 10,000-field
//! assertion. `Display` is Java's `toString()`: the fields asked for so far
//! in `HashMap` order, the leaf size and heap budget.
//! Merging segments that were all written with plain `Lucene90PointsReader`s
//! keeps Java's dispatch only up to the writer: this writer always uses
//! `writeField`, which Java's merge reaches whenever a source was written
//! by an asserting codec (every segment this codec writes). Non-`Asserting`
//! `Lucene104` groups are also checked by the asserting postings consumer;
//! Java checks only the `Asserting` candidate's.

use std::sync::{Arc, Mutex, PoisonError};

use lucene_codecs::bkd_writer::SplitOverride;
use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_codecs::points;
use lucene_codecs::points_writer::{
    AddFinishPolicy, NativePointsWriter, PointsWriteSink, PointsWriteState, PointsWriterFactory,
};
use lucene_codecs::postings_writer;
use lucene_codecs::terms_dictionary::TermsDictionaryFormat;
use lucene_util::java_random::JavaRandom;

use super::asserting_installed::{InstalledAssertingCodec, NAME as ASSERTING};
use super::asserting_points::AssertingPointsWriter;
use super::blockterms::BlockTermsFormat;
use super::bloom::TestBloomFilteredLucenePostings;
use super::mockrandom::MockRandomPostingsFormat;
use crate::index::random_codec::{
    java_string_hash, DocValuesFormat, Error, PostingsFormat, RandomCodec,
};

impl RandomCodec {
    /// This codec as IndexWriter installs it. Every candidate's native
    /// format is built now, as Java's constructor builds them, and lives as
    /// long as the process (terms-dictionary formats are leaked).
    pub fn installed(&self) -> postings_writer::Result<InstalledAssertingCodec> {
        super::register_codecs();
        let postings = self
            .postings_formats()
            .iter()
            .enumerate()
            .map(|(i, format)| Ok(native_postings(format)?.as_instance(instance(i))))
            .collect::<postings_writer::Result<Vec<_>>>()?;
        let doc_values: Vec<_> = self
            .doc_values_formats()
            .iter()
            .enumerate()
            .map(|(i, format)| native_doc_values(format).as_instance(instance(i)))
            .collect();
        let seed = self.per_field_seed();
        let settings = self.points_format();
        let points = RandomPointsFormat(Arc::new(RandomPoints {
            max_points_in_leaf_node: settings.max_points_in_leaf_node,
            max_mb_sort_in_heap: settings.max_mb_sort_in_heap,
            bkd_split_random_seed: settings.bkd_split_random_seed,
        }));
        let postings_names: Vec<String> = self
            .postings_formats()
            .iter()
            .map(|f| postings_display(f))
            .collect();
        let doc_values_names: Vec<String> = self
            .doc_values_formats()
            .iter()
            .map(|f| format!("DocValuesFormat(name={})", f.name()))
            .collect();
        let mappings = Arc::new(Mutex::new(Mappings::default()));
        let (postings_mappings, doc_values_mappings, display_mappings) =
            (mappings.clone(), mappings.clone(), mappings);
        Ok(InstalledAssertingCodec::new()
            .with_postings_format(move |field| {
                let i = pick(seed, field, postings.len());
                lock(&postings_mappings).postings.record(field, i);
                postings[i]
            })
            .with_doc_values_format(move |field| {
                let i = pick(seed, field, doc_values.len());
                lock(&doc_values_mappings).doc_values.record(field, i);
                doc_values[i]
            })
            .with_points_format(Arc::new(points))
            .with_display(move || {
                let mappings = lock(&display_mappings);
                format!(
                    "{ASSERTING}(Lucene104): {}, docValues:{}, maxPointsInLeafNode={}, \
                     maxMBSortInHeap={:?}",
                    mappings.postings.display(&postings_names),
                    mappings.doc_values.display(&doc_values_names),
                    settings.max_points_in_leaf_node,
                    settings.max_mb_sort_in_heap,
                )
            }))
    }
}

/// The candidate's `toString()`.
fn postings_display(format: &PostingsFormat) -> String {
    match format {
        PostingsFormat::Fst => "FST50".to_string(),
        PostingsFormat::Bloom => "TestBloomFilteredLucenePostings(BloomFilteringPostingsFormat(\
                                  PostingsFormat(name=Lucene104)))"
            .to_string(),
        // `BlockTreeOrdsPostingsFormat.toString()` prints a constant.
        PostingsFormat::BlockTreeOrds { .. } => "BlockTreeOrds(blocksize=128)".to_string(),
        other => format!("PostingsFormat(name={})", other.name()),
    }
}

fn lock(mappings: &Mutex<Mappings>) -> std::sync::MutexGuard<'_, Mappings> {
    mappings.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `previousMappings` and `previousDVMappings`: each field's candidate, in
/// the order fields were first asked for.
#[derive(Default)]
struct Mappings {
    postings: FieldMap,
    doc_values: FieldMap,
}

#[derive(Default)]
struct FieldMap(Vec<(String, usize)>);

impl FieldMap {
    fn record(&mut self, field: &str, index: usize) {
        if !self.0.iter().any(|(f, _)| f == field) {
            self.0.push((field.to_string(), index));
            assert!(self.0.len() < 10_000, "test went insane");
        }
    }

    /// `HashMap.toString()`: `{field=format, ...}` in the table's order --
    /// bucket `(h ^ h >>> 16) & (capacity - 1)` of the 16-slot table doubled
    /// past three quarters full, insertion order within a bucket.
    fn display(&self, names: &[String]) -> String {
        let mut capacity = 16usize;
        while self.0.len() * 4 > capacity * 3 {
            capacity *= 2;
        }
        let bucket = |field: &str| {
            let units: Vec<u16> = field.encode_utf16().collect();
            let h = java_string_hash(&units) as u32;
            ((h ^ (h >> 16)) as usize) & (capacity - 1)
        };
        let mut entries: Vec<(usize, &(String, usize))> =
            self.0.iter().map(|e| (bucket(&e.0), e)).collect();
        entries.sort_by_key(|&(b, _)| b);
        let body: Vec<String> = entries
            .iter()
            .map(|(_, (field, i))| format!("{field}={}", names[*i]))
            .collect();
        format!("{{{}}}", body.join(", "))
    }
}

fn instance(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

/// `Math.abs(perFieldSeed ^ field.hashCode()) % formats.size()`.
fn pick(per_field_seed: i32, field: &str, len: usize) -> usize {
    let units: Vec<u16> = field.encode_utf16().collect();
    let hash = (per_field_seed ^ java_string_hash(&units)).wrapping_abs();
    let len = i32::try_from(len).unwrap_or(i32::MAX);
    if len == 0 {
        panic!("{}", Error::EmptyFormats);
    }
    usize::try_from(hash % len).unwrap_or_else(|_| panic!("{}", Error::NegativeFormatIndex))
}

fn leak(format: impl TermsDictionaryFormat + 'static) -> &'static dyn TermsDictionaryFormat {
    Box::leak(Box::new(format))
}

fn native_postings(format: &PostingsFormat) -> postings_writer::Result<Lucene104PostingsFormat> {
    let items = |n: i32| usize::try_from(n).unwrap_or(0);
    Ok(match *format {
        PostingsFormat::Lucene104 {
            min_items,
            max_items,
        } => Lucene104PostingsFormat::new(items(min_items), items(max_items))?,
        PostingsFormat::Fst => Lucene104PostingsFormat::dictionary(
            lucene_codecs::terms_dictionary::lookup(lucene_codecs::memory::FstPostingsFormat::NAME)
                .ok_or_else(|| postings_writer::Error::Dictionary("no FST50 format".into()))?,
        ),
        // `DirectPostingsFormat` writes its `Lucene104` delegate's files;
        // its settings only shape what a reader keeps in memory.
        PostingsFormat::Direct { .. } => Lucene104PostingsFormat::default().named("Direct"),
        PostingsFormat::Bloom => {
            Lucene104PostingsFormat::dictionary(TestBloomFilteredLucenePostings::dictionary_format())
        }
        PostingsFormat::MockRandom { seed } => Lucene104PostingsFormat::dictionary(
            MockRandomPostingsFormat::from_seed(seed).into_dictionary_format(),
        ),
        PostingsFormat::BlockTreeOrds {
            min_items,
            max_items,
        } => Lucene104PostingsFormat::block_tree_ords(items(min_items), items(max_items))?,
        PostingsFormat::FixedGap {
            term_index_interval,
        } => Lucene104PostingsFormat::dictionary(leak(BlockTermsFormat::FixedGap {
            term_index_interval,
        })),
        PostingsFormat::VarGapFixedInterval {
            term_index_interval,
        } => Lucene104PostingsFormat::dictionary(leak(BlockTermsFormat::VarGapFixedInterval {
            term_index_interval,
        })),
        PostingsFormat::VarGapDocFreqInterval {
            term_index_interval,
            doc_freq_threshold,
        } => Lucene104PostingsFormat::dictionary(leak(BlockTermsFormat::VarGapDocFreqInterval {
            doc_freq_threshold,
            term_index_interval,
        })),
        PostingsFormat::Asserting => Lucene104PostingsFormat::default().named(ASSERTING),
    })
}

fn native_doc_values(format: &DocValuesFormat) -> Lucene90DocValuesFormat {
    match *format {
        // Both Lucene90 candidates are the default interval.
        DocValuesFormat::Lucene90 { .. } => Lucene90DocValuesFormat::default(),
        DocValuesFormat::Asserting => Lucene90DocValuesFormat::default().named(ASSERTING),
    }
}

/// `RandomCodec.pointsFormat()`'s `Lucene90PointsWriter` subclass.
#[derive(Debug)]
struct RandomPoints {
    max_points_in_leaf_node: i32,
    max_mb_sort_in_heap: f64,
    bkd_split_random_seed: i32,
}

impl AddFinishPolicy for RandomPoints {
    fn max_points_in_leaf_node(&self) -> i32 {
        self.max_points_in_leaf_node
    }

    fn max_mb_sort_in_heap(&self) -> f64 {
        self.max_mb_sort_in_heap
    }

    /// `RandomlySplittingBKDWriter.split`: `random.nextInt(numIndexDims)`,
    /// from `new Random(bkdSplitRandomSeed ^ fieldInfo.name.hashCode())`.
    fn split(&self, field_name: &str) -> Option<SplitOverride<'static>> {
        let units: Vec<u16> = field_name.encode_utf16().collect();
        let mut random = JavaRandom::new(i64::from(
            self.bkd_split_random_seed ^ java_string_hash(&units),
        ));
        Some(Box::new(
            move |_min: &[u8], _max: &[u8], parent_splits: &[i32]| {
                let dims = i32::try_from(parent_splits.len()).unwrap_or(i32::MAX);
                usize::try_from(random.next_int_bounded(dims)).unwrap_or(0)
            },
        ))
    }
}

/// `new AssertingPointsFormat(<the subclass's PointsFormat>)`.
#[derive(Debug)]
struct RandomPointsFormat(Arc<RandomPoints>);

impl PointsWriterFactory for RandomPointsFormat {
    fn fields_writer<'a>(
        &self,
        state: PointsWriteState<'a>,
    ) -> points::Result<Box<dyn PointsWriteSink + 'a>> {
        let policy: Arc<dyn AddFinishPolicy> = self.0.clone();
        Ok(Box::new(AssertingPointsWriter::new(
            NativePointsWriter::with_add_finish(state, policy)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new RandomCodec(new Random(seed))` asked for fields `a`..`p` (postings
    /// and doc values each), then `toString()`, as Lucene prints it.
    #[test]
    fn display_is_javas_to_string() {
        for (seed, expected) in [
            (
                5,
                "Asserting(Lucene104): {a=BlockTreeOrds(blocksize=128), \
             b=PostingsFormat(name=Asserting), c=PostingsFormat(name=MockRandom), \
             d=PostingsFormat(name=LuceneVarGapDocFreqInterval), \
             e=BlockTreeOrds(blocksize=128), f=PostingsFormat(name=Asserting), \
             g=PostingsFormat(name=MockRandom), \
             h=PostingsFormat(name=LuceneVarGapDocFreqInterval), \
             i=BlockTreeOrds(blocksize=128), j=PostingsFormat(name=Asserting), \
             k=PostingsFormat(name=MockRandom), \
             l=PostingsFormat(name=LuceneVarGapDocFreqInterval), \
             m=BlockTreeOrds(blocksize=128), n=PostingsFormat(name=Asserting), \
             o=PostingsFormat(name=MockRandom), \
             p=PostingsFormat(name=LuceneVarGapDocFreqInterval)}, \
             docValues:{a=DocValuesFormat(name=Lucene90), b=DocValuesFormat(name=Lucene90), \
             c=DocValuesFormat(name=Asserting), d=DocValuesFormat(name=Lucene90), \
             e=DocValuesFormat(name=Lucene90), f=DocValuesFormat(name=Asserting), \
             g=DocValuesFormat(name=Lucene90), h=DocValuesFormat(name=Lucene90), \
             i=DocValuesFormat(name=Asserting), j=DocValuesFormat(name=Lucene90), \
             k=DocValuesFormat(name=Lucene90), l=DocValuesFormat(name=Asserting), \
             m=DocValuesFormat(name=Lucene90), n=DocValuesFormat(name=Lucene90), \
             o=DocValuesFormat(name=Asserting), p=DocValuesFormat(name=Lucene90)}, \
             maxPointsInLeafNode=134, maxMBSortInHeap=7.834923958562488",
            ),
            (
                14,
                "Asserting(Lucene104): {a=PostingsFormat(name=LuceneFixedGap), b=FST50, \
             c=PostingsFormat(name=LuceneVarGapFixedInterval), \
             d=PostingsFormat(name=Lucene104), e=PostingsFormat(name=LuceneFixedGap), \
             f=FST50, g=PostingsFormat(name=LuceneVarGapFixedInterval), \
             h=PostingsFormat(name=Lucene104), i=PostingsFormat(name=LuceneFixedGap), \
             j=FST50, k=PostingsFormat(name=LuceneVarGapFixedInterval), \
             l=PostingsFormat(name=Lucene104), m=PostingsFormat(name=LuceneFixedGap), \
             n=FST50, o=PostingsFormat(name=LuceneVarGapFixedInterval), \
             p=PostingsFormat(name=Lucene104)}, docValues:{a=DocValuesFormat(name=Lucene90), \
             b=DocValuesFormat(name=Asserting), c=DocValuesFormat(name=Lucene90), \
             d=DocValuesFormat(name=Lucene90), e=DocValuesFormat(name=Asserting), \
             f=DocValuesFormat(name=Lucene90), g=DocValuesFormat(name=Lucene90), \
             h=DocValuesFormat(name=Asserting), i=DocValuesFormat(name=Lucene90), \
             j=DocValuesFormat(name=Lucene90), k=DocValuesFormat(name=Asserting), \
             l=DocValuesFormat(name=Lucene90), m=DocValuesFormat(name=Lucene90), \
             n=DocValuesFormat(name=Asserting), o=DocValuesFormat(name=Lucene90), \
             p=DocValuesFormat(name=Lucene90)}, maxPointsInLeafNode=34, \
             maxMBSortInHeap=6.7426961678223645",
            ),
        ] {
            let codec = RandomCodec::new(&mut JavaRandom::new(seed), false, 1)
                .installed()
                .unwrap();
            use lucene_codecs::codec::Lucene104Codec;
            for field in "abcdefghijklmnop".chars().map(String::from) {
                codec.postings_format_for_field(&field);
                codec.doc_values_format_for_field(&field);
            }
            assert_eq!(codec.to_string(), expected, "seed {seed}");
        }
    }
}
