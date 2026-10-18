//! `SpellChecker` (M13 T13.3e): the classic spell checker, which keeps its
//! own index of dictionary words, each with its character n-grams (`gramN`)
//! and its first and last n-gram (`startN`, `endN`). A misspelled word's
//! n-grams are searched as one disjunction (BM25, the first n-gram boosted by
//! 2 and the last by 1), and the hits are re-ranked by a [`StringDistance`].
//!
//! The spell index is a real Lucene index written by this port's
//! `IndexWriter`, so Lucene can read it and the BM25 scores are Lucene's.
//!
//! Departures from Java: the checker borrows its [`Directory`]; there is no
//! `IndexWriterConfig` argument (the writer's defaults are used, and
//! `clearIndex` deletes every document where Java recreates the index); the
//! searcher is swapped without Java's locks, `&mut self` serialising the
//! changes instead. `numSug` 0 is refused, as Java's searcher refuses 0 hits.

use std::sync::Arc;

use lucene_index::document::{Document, Field, FieldType, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_store::Directory;

use super::distance::{LevenshteinDistance, StringDistance};
use super::suggest_word::{
    score_comparator, SuggestMode, SuggestWord, SuggestWordComparator, SuggestWordQueue,
};
use super::Dictionary;
use crate::directory_reader::DirectoryReader;
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::query::{BooleanQuery, BoostQuery, Clause, TermQuery};
use crate::reader::IndexReader;
use crate::{Error, Result};

/// `SpellChecker.DEFAULT_ACCURACY`.
pub const DEFAULT_ACCURACY: f32 = 0.5;

/// `SpellChecker.F_WORD`: the stored field holding each dictionary word.
pub const F_WORD: &str = "word";

fn index_error(e: lucene_index::index_writer::Error) -> Error {
    Error::IllegalState(e.to_string())
}

/// `getMin(l)`: the shortest n-gram for a word of `l` UTF-16 units.
fn get_min(l: usize) -> usize {
    match l {
        l if l > 5 => 3,
        5 => 2,
        _ => 1,
    }
}

/// `getMax(l)`: the longest n-gram for a word of `l` UTF-16 units.
fn get_max(l: usize) -> usize {
    match l {
        l if l > 5 => 4,
        5 => 3,
        l => l.min(2),
    }
}

/// `formGrams(text, ng)`: every `ng`-unit substring, in order, duplicates
/// kept. Java cuts UTF-16 units, so a gram may split a surrogate pair; its
/// lone half becomes U+FFFD, as Java's UTF-8 encoding of the term does.
fn form_grams(text: &[u16], ng: usize) -> Vec<String> {
    // `ng` is at least 1 here: `getMin` never answers 0
    text.windows(ng.max(1))
        .map(String::from_utf16_lossy)
        .collect()
}

/// `createDocument(text, ng1, ng2)` with `addGram`.
fn create_document(text: &str, ng1: usize, ng2: usize) -> Result<Document> {
    let mut doc = Document::new();
    doc.add(StringField::new(F_WORD, text, Store::Yes));
    let units: Vec<u16> = text.encode_utf16().collect();
    let mut gram_type = FieldType::copy_of(&StringField::type_not_stored());
    gram_type
        .set_index_options(lucene_codecs::field_infos::IndexOptions::DocsAndFreqs)
        .map_err(|e| Error::IllegalState(e.to_string()))?;
    gram_type.freeze();
    for ng in ng1..=ng2 {
        let key = format!("gram{ng}");
        let grams = form_grams(&units, ng);
        for (i, gram) in grams.iter().enumerate() {
            doc.add(
                Field::from_string(key.clone(), gram.clone(), FieldType::copy_of(&gram_type))
                    .map_err(|e| Error::IllegalState(e.to_string()))?,
            );
            if i == 0 {
                doc.add(StringField::new(
                    format!("start{ng}"),
                    gram.clone(),
                    Store::No,
                ));
            }
        }
        // may not be present if len == ng1
        if let Some(end) = grams.last() {
            doc.add(StringField::new(format!("end{ng}"), end.clone(), Store::No));
        }
    }
    Ok(doc)
}

/// `DirectoryReader.indexExists`: the directory holds a commit.
fn index_exists(dir: &dyn Directory) -> bool {
    lucene_index::segment_infos::read_latest(dir).is_ok()
}

/// `SpellChecker`.
pub struct SpellChecker<'d> {
    spell_index: &'d dyn Directory,
    b_start: f32,
    b_end: f32,
    /// The searcher's reader; `None` once closed.
    reader: Option<DirectoryReader>,
    accuracy: f32,
    sd: Arc<dyn StringDistance>,
    comparator: SuggestWordComparator,
}

impl std::fmt::Debug for SpellChecker<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpellChecker")
            .field("accuracy", &self.accuracy)
            .field("sd", &self.sd)
            .field("closed", &self.reader.is_none())
            .finish_non_exhaustive()
    }
}

impl<'d> SpellChecker<'d> {
    /// `new SpellChecker(spellIndex)`: Levenshtein distance, the score
    /// comparator.
    ///
    /// # Errors
    /// The spell index cannot be created or opened.
    pub fn new(spell_index: &'d dyn Directory) -> Result<Self> {
        Self::with_distance(spell_index, Arc::new(LevenshteinDistance::new()))
    }

    /// `new SpellChecker(spellIndex, sd)`.
    ///
    /// # Errors
    /// The spell index cannot be created or opened.
    pub fn with_distance(
        spell_index: &'d dyn Directory,
        sd: Arc<dyn StringDistance>,
    ) -> Result<Self> {
        Self::with_comparator(spell_index, sd, score_comparator())
    }

    /// `new SpellChecker(spellIndex, sd, comparator)`.
    ///
    /// # Errors
    /// The spell index cannot be created or opened.
    pub fn with_comparator(
        spell_index: &'d dyn Directory,
        sd: Arc<dyn StringDistance>,
        comparator: SuggestWordComparator,
    ) -> Result<Self> {
        let mut s = Self {
            spell_index,
            b_start: 2.0,
            b_end: 1.0,
            reader: None,
            accuracy: DEFAULT_ACCURACY,
            sd,
            comparator,
        };
        s.open_spell_index(spell_index)?;
        Ok(s)
    }

    /// `setSpellIndex(spellIndexDir)`: use another spell index, created
    /// empty if it has no commit.
    ///
    /// # Errors
    /// The checker is closed, or the index cannot be created or opened.
    pub fn set_spell_index(&mut self, spell_index: &'d dyn Directory) -> Result<()> {
        self.ensure_open()?;
        self.open_spell_index(spell_index)
    }

    fn open_spell_index(&mut self, spell_index: &'d dyn Directory) -> Result<()> {
        if !index_exists(spell_index) {
            let mut writer = IndexWriter::open(spell_index, Vec::new(), "Lucene104", LATEST)
                .map_err(index_error)?;
            writer.commit().map_err(index_error)?;
        }
        self.swap_searcher(spell_index)
    }

    /// `setComparator`.
    pub fn set_comparator(&mut self, comparator: SuggestWordComparator) {
        self.comparator = comparator;
    }

    /// `getComparator`.
    pub fn get_comparator(&self) -> SuggestWordComparator {
        self.comparator.clone()
    }

    /// `setStringDistance`.
    pub fn set_string_distance(&mut self, sd: Arc<dyn StringDistance>) {
        self.sd = sd;
    }

    /// `getStringDistance`.
    pub fn get_string_distance(&self) -> Arc<dyn StringDistance> {
        self.sd.clone()
    }

    /// `setAccuracy`.
    pub fn set_accuracy(&mut self, accuracy: f32) {
        self.accuracy = accuracy;
    }

    /// `getAccuracy`.
    pub fn get_accuracy(&self) -> f32 {
        self.accuracy
    }

    /// `suggestSimilar(word, numSug)`: suggestions from the spell index
    /// alone, the best first.
    ///
    /// # Errors
    /// The checker is closed, `num_sug` is 0, or the search fails.
    pub fn suggest_similar(&self, word: &str, num_sug: usize) -> Result<Vec<String>> {
        self.suggest_similar_in_with_accuracy(
            word,
            num_sug,
            None,
            SuggestMode::SuggestWhenNotInIndex,
            self.accuracy,
        )
    }

    /// `suggestSimilar(word, numSug, accuracy)`.
    ///
    /// # Errors
    /// As [`Self::suggest_similar`].
    pub fn suggest_similar_with_accuracy(
        &self,
        word: &str,
        num_sug: usize,
        accuracy: f32,
    ) -> Result<Vec<String>> {
        self.suggest_similar_in_with_accuracy(
            word,
            num_sug,
            None,
            SuggestMode::SuggestWhenNotInIndex,
            accuracy,
        )
    }

    /// `suggestSimilar(word, numSug, ir, field, suggestMode)`: with a user
    /// index (`ir` and `field`), a suggestion must occur in it, and the mode
    /// decides whether a word found there is answered as is.
    ///
    /// # Errors
    /// As [`Self::suggest_similar`], or reading `ir` fails.
    pub fn suggest_similar_in(
        &self,
        word: &str,
        num_sug: usize,
        user_index: Option<(&dyn IndexReader, &str)>,
        suggest_mode: SuggestMode,
    ) -> Result<Vec<String>> {
        self.suggest_similar_in_with_accuracy(
            word,
            num_sug,
            user_index,
            suggest_mode,
            self.accuracy,
        )
    }

    /// `suggestSimilar(word, numSug, ir, field, suggestMode, accuracy)`.
    ///
    /// # Errors
    /// As [`Self::suggest_similar_in`].
    pub fn suggest_similar_in_with_accuracy(
        &self,
        word: &str,
        num_sug: usize,
        user_index: Option<(&dyn IndexReader, &str)>,
        suggest_mode: SuggestMode,
        mut accuracy: f32,
    ) -> Result<Vec<String>> {
        let reader = self.searcher_reader()?;
        // a missing reader or field: SUGGEST_ALWAYS, and SUGGEST_ALWAYS
        // forgets them
        let suggest_mode = if user_index.is_none() {
            SuggestMode::SuggestAlways
        } else {
            suggest_mode
        };
        let user_index = if suggest_mode == SuggestMode::SuggestAlways {
            None
        } else {
            user_index
        };
        let units: Vec<u16> = word.encode_utf16().collect();
        let length_word = units.len();
        let freq = match user_index {
            Some((ir, field)) => ir.doc_freq(field, word.as_bytes())? as i32,
            None => 0,
        };
        let goal_freq = if suggest_mode == SuggestMode::SuggestMorePopular {
            freq
        } else {
            0
        };
        // if the word exists in the real index and we don't care for word
        // frequency, return the word itself
        if suggest_mode == SuggestMode::SuggestWhenNotInIndex && freq > 0 {
            return Ok(vec![word.to_string()]);
        }

        let mut query = BooleanQuery::new();
        for ng in get_min(length_word)..=get_max(length_word) {
            let key = format!("gram{ng}");
            let grams = form_grams(&units, ng);
            // every n-gram size fits the word, so there is a first and a last
            if let (Some(first), Some(last)) = (grams.first(), grams.last()) {
                if self.b_start > 0.0 {
                    query.should.push(Clause::from(BoostQuery::new(
                        TermQuery::new(format!("start{ng}"), first.as_bytes()),
                        self.b_start,
                    )));
                }
                if self.b_end > 0.0 {
                    query.should.push(Clause::from(BoostQuery::new(
                        TermQuery::new(format!("end{ng}"), last.as_bytes()),
                        self.b_end,
                    )));
                }
            }
            for gram in &grams {
                query
                    .should
                    .push(Clause::from(TermQuery::new(key.clone(), gram.as_bytes())));
            }
        }

        if num_sug == 0 {
            return Err(Error::IllegalArgument("numHits must be > 0".into()));
        }
        let max_hits = num_sug.saturating_mul(10);
        let hits = Self::search(reader, query, max_hits)?;
        let mut sug_queue = SuggestWordQueue::with_comparator(num_sug, self.comparator.clone());
        let segments = reader.segment_readers();
        let word_numbers: Vec<Option<i32>> = segments
            .iter()
            .map(|s| s.field_infos().field_by_name(F_WORD).map(|f| f.number))
            .collect();
        for (doc, _) in hits.into_iter().take(max_hits) {
            let Some(string) = Self::stored_word(reader, &word_numbers, doc)? else {
                continue;
            };
            // don't suggest a word for itself, that would be silly
            if string == word {
                continue;
            }
            let score = self.sd.get_distance(word, &string);
            if score < accuracy {
                continue;
            }
            let mut freq = 0;
            if let Some((ir, field)) = user_index {
                // use the user index: the word's frequency there
                freq = ir.doc_freq(field, string.as_bytes())? as i32;
                // don't suggest a word that is not present in the field
                if (suggest_mode == SuggestMode::SuggestMorePopular && goal_freq > freq) || freq < 1
                {
                    continue;
                }
            }
            sug_queue.insert_with_overflow(SuggestWord {
                score,
                freq,
                string,
            });
            if sug_queue.size() == num_sug {
                // if the queue is full, maintain the minScore score
                accuracy = sug_queue.top().map_or(accuracy, |w| w.score);
            }
        }
        // convert to array of strings, the best first
        let mut list = Vec::with_capacity(sug_queue.size());
        while let Some(w) = sug_queue.pop() {
            list.push(w.string);
        }
        list.reverse();
        Ok(list)
    }

    /// `indexSearcher.search(query, maxHits).scoreDocs` as `(doc, score)`:
    /// the query rewritten first, as `IndexSearcher.search` does.
    fn search(
        reader: &DirectoryReader,
        query: BooleanQuery,
        max_hits: usize,
    ) -> Result<Vec<(i32, f32)>> {
        let query = match Clause::from(query).rewrite() {
            Clause::MatchNoDocs(_) => return Ok(Vec::new()),
            Clause::Boolean(b) => *b,
            other => BooleanQuery::new().with_must([other]),
        };
        let opened = reader.open_segments()?;
        let segments = opened.as_open_segments();
        // the gram fields are `StringField`s, which omit norms
        let owned = reader.field_norms_by_field(&[]);
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segments, &norms)?;
        let top = searcher.search(&query, max_hits)?;
        Ok(top.score_docs.iter().map(|d| (d.doc, d.score)).collect())
    }

    /// `storedFields.document(doc).get(F_WORD)`.
    fn stored_word(
        reader: &DirectoryReader,
        word_numbers: &[Option<i32>],
        doc: i32,
    ) -> Result<Option<String>> {
        let mut base = 0i32;
        for (seg, number) in reader.segment_readers().iter().zip(word_numbers) {
            if doc >= base + seg.max_doc {
                base += seg.max_doc;
                continue;
            }
            let stored = match number {
                Some(_) => seg.stored_document(doc - base)?,
                None => None,
            };
            let field = stored.and_then(|d| {
                d.fields
                    .into_iter()
                    .find(|f| Some(f.field_number) == *number)
            });
            return Ok(field.and_then(|f| match f.value {
                lucene_codecs::stored_fields::FieldValue::String(s) => Some(s),
                _ => None,
            }));
        }
        Ok(None)
    }

    /// `clearIndex()`: every word removed from the spell index.
    ///
    /// # Errors
    /// The checker is closed, or the index cannot be written.
    pub fn clear_index(&mut self) -> Result<()> {
        self.ensure_open()?;
        let dir = self.spell_index;
        {
            let mut writer =
                IndexWriter::open(dir, Vec::new(), "Lucene104", LATEST).map_err(index_error)?;
            writer.delete_all().map_err(index_error)?;
            writer.commit().map_err(index_error)?;
        }
        self.swap_searcher(dir)
    }

    /// `exist(word)`: whether the spell index holds `word`.
    ///
    /// # Errors
    /// The checker is closed, or the index cannot be read.
    pub fn exist(&self, word: &str) -> Result<bool> {
        Ok(self.searcher_reader()?.doc_freq(F_WORD, word.as_bytes())? > 0)
    }

    /// `indexDictionary(dict, config, fullMerge)`: adds every word of
    /// `dict` of three or more UTF-16 units that the spell index does not
    /// already hold, then merges to one segment if asked.
    ///
    /// # Errors
    /// The checker is closed, or the dictionary or index fails.
    pub fn index_dictionary(&mut self, dict: &dyn Dictionary, full_merge: bool) -> Result<()> {
        self.ensure_open()?;
        let dir = self.spell_index;
        {
            let reader = self.searcher_reader()?;
            let leaves = if IndexReader::max_doc(reader) > 0 {
                reader.leaves()
            } else {
                Vec::new()
            };
            let mut terms = Vec::new();
            for ctx in &leaves {
                terms.extend(ctx.reader.terms(F_WORD)?);
            }
            let mut terms_enums = terms
                .iter()
                .map(|t| t.iterator())
                .collect::<Result<Vec<_>>>()?;
            let mut writer =
                IndexWriter::open(dir, Vec::new(), "Lucene104", LATEST).map_err(index_error)?;
            let mut iter = dict.get_entry_iterator()?;
            'terms: while let Some(current_term) = iter.next() {
                let word = String::from_utf8_lossy(&current_term).into_owned();
                let len = word.encode_utf16().count();
                if len < 3 {
                    // too short we bail but "too long" is fine...
                    continue;
                }
                for te in &mut terms_enums {
                    // the word is already in the index
                    if te.try_seek_exact(&current_term)? {
                        continue 'terms;
                    }
                }
                let doc = create_document(&word, get_min(len), get_max(len))?;
                writer.add_fields_document(&doc).map_err(index_error)?;
            }
            writer.commit().map_err(index_error)?;
            if full_merge {
                writer.force_merge(1).map_err(index_error)?;
                writer.commit().map_err(index_error)?;
            }
        }
        // also re-open the spell index to see our own changes when the next
        // suggestion is fetched
        self.swap_searcher(dir)
    }

    /// `close()`.
    ///
    /// # Errors
    /// Already closed.
    pub fn close(&mut self) -> Result<()> {
        self.ensure_open()?;
        self.reader = None;
        Ok(())
    }

    /// `isClosed()`.
    pub fn is_closed(&self) -> bool {
        self.reader.is_none()
    }

    fn ensure_open(&self) -> Result<()> {
        self.searcher_reader().map(|_| ())
    }

    fn searcher_reader(&self) -> Result<&DirectoryReader> {
        self.reader
            .as_ref()
            .ok_or_else(|| Error::AlreadyClosed("Spellchecker has been closed".into()))
    }

    fn swap_searcher(&mut self, dir: &'d dyn Directory) -> Result<()> {
        let reader = DirectoryReader::open(dir)?;
        self.reader = Some(reader);
        self.spell_index = dir;
        Ok(())
    }
}
