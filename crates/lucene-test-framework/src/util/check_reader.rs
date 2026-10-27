//! `TestUtil.checkReader`: `CheckIndex`'s per-segment tests run against the
//! reader API, so a view (a filter, a sorting or slow-codec wrapper, a
//! test-framework asserting reader) is checked the way its consumers read
//! it, not through the files behind it.
//!
//! Each leaf gets Java's `MIN_LEVEL_FOR_SLOW_CHECKS` order: `checkIntegrity`,
//! live docs, postings (`checkFields`: field flags against the field infos,
//! min/max terms, term order, ordinals, every posting's documents,
//! frequencies, positions, payloads and offsets, `nextPostings` against
//! `nextDoc`, term and field statistics, norms against the visited
//! documents, the last-term seek, the seven `advance` probes and the
//! impacts walk checked by [`check_impacts`]), stored fields, term vectors
//! (each vector term's postings matched against the index's, the slow
//! level's cross-check), doc values (each iterator against a second one by
//! `advanceExact`, ordinals and lookups, the bulk `longValues` fetch),
//! points and vectors, then `checkReaderSanity`.
//!
//! Rust's shape changes three things. A `FieldInfos` is borrowed, so Java's
//! same-instance check has no counterpart; `docIDRunEnd` is not on the
//! reader API's postings, so `checkDocIDRuns` is the file `CheckIndex`'s
//! alone; and points are walked with a verifying visitor (counts, distinct
//! documents, bounds, one-dimension order) instead of `PointTree`'s
//! node-by-node checks, which the reader API does not expose.

use lucene_codecs::blocktree::SeekStatus;
use lucene_codecs::field_infos::{DocValuesType, FieldInfo, IndexOptions};
use lucene_index::check_index::check_impacts;
use lucene_search::reader::{
    DocAndFloatFeatureBuffer, IndexReader, IntersectVisitor, LeafReader, PostingsFlags, Relation,
    StoredFieldVisitor, Terms, VisitStatus, NO_MORE_DOCS,
};
use lucene_util::fixed_bit_set::FixedBitSet;

/// What a check fails with: Java's `CheckIndexException` message.
type Check = Result<(), String>;

fn err<T>(e: impl std::fmt::Display) -> Result<T, String> {
    Err(e.to_string())
}

macro_rules! io {
    ($e:expr) => {
        $e.map_err(|e| e.to_string())?
    };
}

/// `IndexWriter.MAX_POSITION`.
const MAX_POSITION: i32 = i32::MAX - 128;

/// `checkReader(IndexReader)`: every leaf at the slow-checks level.
///
/// # Errors
/// The first failed check, with Java's message.
pub fn check_reader(reader: &dyn IndexReader) -> Check {
    for leaf in reader.leaves() {
        check_leaf_reader(leaf.reader, true)?;
    }
    Ok(())
}

/// `checkReader(LeafReader, level)`: `slow` is a level of at least
/// `MIN_LEVEL_FOR_SLOW_CHECKS`.
///
/// # Errors
/// The first failed check, with Java's message.
pub fn check_leaf_reader(reader: &dyn LeafReader, slow: bool) -> Check {
    io!(reader.check_integrity());
    test_live_docs(reader)?;
    test_postings(reader, slow)?;
    test_stored_fields(reader)?;
    test_term_vectors(reader, slow)?;
    test_doc_values(reader)?;
    test_points(reader)?;
    test_vectors(reader)?;
    check_reader_sanity(reader)
}

fn idx(doc: i32) -> usize {
    usize::try_from(doc).unwrap_or(usize::MAX)
}

fn max_doc_bits(reader: &dyn LeafReader) -> FixedBitSet {
    FixedBitSet::new(idx(reader.max_doc()))
}

fn is_live(reader: &dyn LeafReader, doc: i32) -> bool {
    reader
        .live_docs()
        .is_none_or(|bits| idx(doc) < bits.len() && bits.get(idx(doc)))
}

/// `BytesRef.toString()`: `[61 62]`.
fn bytes(term: &[u8]) -> String {
    let hex: Vec<String> = term.iter().map(|b| format!("{b:x}")).collect();
    format!("[{}]", hex.join(" "))
}

/// `indexOptions.compareTo(DOCS_AND_FREQS) >= 0`.
fn has_freqs(opts: IndexOptions) -> bool {
    !matches!(opts, IndexOptions::None | IndexOptions::Docs)
}

/// `testLiveDocs`.
fn test_live_docs(reader: &dyn LeafReader) -> Check {
    let max_doc = reader.max_doc();
    let num_docs = reader.num_docs();
    match reader.live_docs() {
        Some(bits) => {
            if bits.len() != idx(max_doc) {
                return err(format!(
                    "liveDocs.length()={} != maxDoc={max_doc}",
                    bits.len()
                ));
            }
            let live = i32::try_from(bits.cardinality()).unwrap_or(i32::MAX);
            if live != num_docs {
                return err(format!(
                    "numDocs={num_docs} but liveDocs has {live} live docs"
                ));
            }
        }
        None if num_docs != max_doc => {
            return err(format!(
                "liveDocs is null but numDocs={num_docs} != maxDoc={max_doc}"
            ));
        }
        None => {}
    }
    Ok(())
}

/// `testPostings` -> `checkFields` over each indexed field.
fn test_postings(reader: &dyn LeafReader, slow: bool) -> Check {
    let mut names: Vec<&FieldInfo> = reader
        .field_infos()
        .fields
        .iter()
        .filter(|f| f.index_options != IndexOptions::None)
        .collect();
    names.sort_by(|a, b| a.name.cmp(&b.name));
    for fi in names {
        if let Some(terms) = io!(reader.terms(&fi.name)) {
            check_field(reader, fi, terms.as_ref(), slow)?;
        }
    }
    Ok(())
}

fn check_field(reader: &dyn LeafReader, fi: &FieldInfo, terms: &dyn Terms, slow: bool) -> Check {
    let field = fi.name.as_str();
    let max_doc = reader.max_doc();
    if terms.doc_count() > max_doc {
        return err(format!(
            "docCount > maxDoc for field: {field}, docCount={}, maxDoc={max_doc}",
            terms.doc_count()
        ));
    }
    let has_freqs = terms.has_freqs();
    let has_positions = terms.has_positions();
    let has_offsets = terms.has_offsets();
    let min_term = io!(terms.min());
    let max_term = io!(terms.max());
    match (&min_term, &max_term) {
        (None, Some(_)) => {
            return err(format!(
                "field \"{field}\" has null minTerm but non-null maxTerm"
            ))
        }
        (Some(_), None) => {
            return err(format!(
                "field \"{field}\" has non-null minTerm but null maxTerm"
            ))
        }
        _ => {}
    }
    let opts = fi.index_options;
    for (name, has, expected) in [
        ("hasFreqs", has_freqs, self::has_freqs(opts)),
        ("hasPositions", has_positions, opts.subsumes_positions()),
        ("hasPayloads", terms.has_payloads(), fi.store_payloads),
        ("hasOffsets", has_offsets, opts.subsumes_offsets()),
    ] {
        if has != expected {
            return err(format!(
                "field \"{field}\" should have {name}={expected} but got {has}"
            ));
        }
    }

    let mut te = io!(terms.iterator());
    let mut has_ord = true;
    let mut term_count = 0i64;
    let mut last_term: Option<Vec<u8>> = None;
    let (mut sum_ttf, mut sum_df) = (0i64, 0i64);
    let mut visited = max_doc_bits(reader);
    while let Some(term) = io!(te.next()) {
        let term = term.to_vec();
        if let Some(last) = &last_term {
            if *last >= term {
                return err(format!(
                    "terms out of order: lastTerm={} term={}",
                    bytes(last),
                    bytes(&term)
                ));
            }
        }
        if let (Some(min), Some(max)) = (&min_term, &max_term) {
            if term < *min {
                return err(format!(
                    "field=\"{field}\": invalid term: term={}, minTerm={}",
                    bytes(&term),
                    bytes(min)
                ));
            }
            if term > *max {
                return err(format!(
                    "field=\"{field}\": invalid term: term={}, maxTerm={}",
                    bytes(&term),
                    bytes(max)
                ));
            }
        } else {
            return err(format!(
                "field=\"{field}\": invalid term: term={}, minTerm=null",
                bytes(&term)
            ));
        }
        let doc_freq = io!(te.doc_freq());
        if doc_freq <= 0 {
            return err(format!("docfreq: {doc_freq} is out of bounds"));
        }
        sum_df = sum_df.saturating_add(i64::from(doc_freq));
        let ttf = io!(te.total_term_freq());
        if !has_freqs && ttf != i64::from(doc_freq) {
            return err(format!(
                "field \"{field}\" hasFreqs is false, but TermsEnum.totalTermFreq()={ttf} \
                 (should be {doc_freq})"
            ));
        }
        if has_ord {
            match te.ord() {
                Ok(ord) if ord != term_count => {
                    return err(format!(
                        "ord mismatch: TermsEnum has ord={ord} vs actual={term_count}"
                    ));
                }
                Ok(_) => {}
                Err(_) => has_ord = false,
            }
        }
        let (doc_count, total_tf) = check_term_postings(
            te.as_mut(),
            &term,
            has_freqs,
            has_positions,
            has_offsets,
            max_doc,
            &mut visited,
        )?;
        if doc_count != doc_freq {
            return err(format!(
                "term {} docFreq={doc_freq} != tot docs w/o deletions {doc_count}",
                bytes(&term)
            ));
        }
        if doc_freq > terms.doc_count() {
            return err(format!(
                "term {} docFreq={doc_freq} > docCount={}",
                bytes(&term),
                terms.doc_count()
            ));
        }
        if ttf <= 0 {
            return err(format!("totalTermFreq: {ttf} is out of bounds"));
        }
        sum_ttf = sum_ttf.saturating_add(total_tf);
        if total_tf != ttf {
            return err(format!(
                "term {} totalTermFreq={ttf} != recomputed totalTermFreq={total_tf}",
                bytes(&term)
            ));
        }
        check_advance(te.as_mut(), &term, has_freqs, has_positions, max_doc)?;
        if slow {
            check_term_impacts(te.as_mut(), &term)?;
        }
        term_count = term_count.saturating_add(1);
        last_term = Some(term);
    }
    if min_term.is_some() && term_count == 0 {
        return err(format!(
            "field=\"{field}\": minTerm is non-null yet we saw no terms: {}",
            min_term.as_deref().map(bytes).unwrap_or_default()
        ));
    }
    if sum_df != terms.sum_doc_freq() {
        return err(format!(
            "sumDocFreq for field {field}={} != recomputed sumDocFreq={sum_df}",
            terms.sum_doc_freq()
        ));
    }
    if sum_ttf != terms.sum_total_term_freq() {
        return err(format!(
            "sumTotalTermFreq for field {field}={} != recomputed sumTotalTermFreq={sum_ttf}",
            terms.sum_total_term_freq()
        ));
    }
    if !has_freqs && sum_ttf != sum_df {
        return err(format!(
            "sumTotalTermFreq for field {field} should be {sum_df}, got sumTotalTermFreq={sum_ttf}"
        ));
    }
    let visited_count = i32::try_from(visited.cardinality()).unwrap_or(i32::MAX);
    if visited_count != terms.doc_count() {
        return err(format!(
            "docCount for field {field}={} != recomputed docCount={visited_count}",
            terms.doc_count()
        ));
    }
    if terms.size() != -1 && terms.size() != term_count {
        return err(format!(
            "termCount mismatch {} vs {term_count}",
            terms.size()
        ));
    }
    if fi.index_options != IndexOptions::None && !fi.omit_norms {
        check_norms_vs_postings(reader, field, &visited)?;
    }
    if let Some(last) = last_term {
        check_last_term(te.as_mut(), &last)?;
    }
    Ok(())
}

/// One term's `PostingsEnum.ALL` walk: documents, frequencies, positions,
/// payloads and offsets, with a second enum's `nextPostings` alongside.
/// Returns the document count and the summed frequency.
fn check_term_postings(
    te: &mut dyn lucene_search::reader::TermsEnum,
    term: &[u8],
    has_freqs: bool,
    has_positions: bool,
    has_offsets: bool,
    max_doc: i32,
    visited: &mut FixedBitSet,
) -> Result<(i32, i64), String> {
    let t = bytes(term);
    let mut postings = io!(te.postings(PostingsFlags::All));
    let mut bulk = io!(te.postings(PostingsFlags::All));
    io!(bulk.next_doc());
    let mut buffer = DocAndFloatFeatureBuffer::default();
    let mut buffer_index = 0usize;
    let (mut last_doc, mut doc_count, mut total_tf) = (-1i32, 0i32, 0i64);
    loop {
        let doc = io!(postings.next_doc());
        if doc == NO_MORE_DOCS {
            break;
        }
        if doc >= 0 && doc < max_doc {
            // FBS: `visited` holds `maxDoc` bits and `doc` was just bounded by it.
            visited.set(idx(doc));
        }
        let freq = postings.freq();
        if freq <= 0 {
            return err(format!("term {t}: doc {doc}: freq {freq} is out of bounds"));
        }
        if buffer_index == buffer.docs.len() {
            let up_to = i32::try_from((i64::from(bulk.doc_id()) + 64).min(i64::from(i32::MAX)))
                .unwrap_or(i32::MAX);
            io!(bulk.next_postings(up_to, &mut buffer));
            buffer_index = 0;
        }
        let Some(&bulk_doc) = buffer.docs.get(buffer_index) else {
            return err(format!("Doc {doc} not found by PostingsEnum#nextPostings"));
        };
        if bulk_doc != doc {
            return err(format!(
                "PostingsEnum#nextPostings returns {bulk_doc} as next doc while \
                 PostingsEnum#nextDoc returns {doc}"
            ));
        }
        let bulk_freq = buffer
            .features
            .get(buffer_index)
            .copied()
            .unwrap_or(f32::NAN);
        // Java compares the int freq with the float feature.
        if f64::from(bulk_freq) != f64::from(freq) {
            return err(format!(
                "PostingsEnum#nextPostings returns {bulk_freq} as term freq while \
                 PostingsEnum#freq returns {freq}"
            ));
        }
        buffer_index = buffer_index.saturating_add(1);
        if !has_freqs && freq != 1 {
            return err(format!(
                "term {t}: doc {doc}: freq {freq} != 1 when Terms.hasFreqs() is false"
            ));
        }
        total_tf = total_tf.saturating_add(i64::from(freq));
        doc_count = doc_count.saturating_add(1);
        if doc <= last_doc {
            return err(format!("term {t}: doc {doc} <= lastDoc {last_doc}"));
        }
        if doc >= max_doc {
            return err(format!("term {t}: doc {doc} >= maxDoc {max_doc}"));
        }
        last_doc = doc;
        if has_positions {
            let (mut last_pos, mut last_offset) = (-1i32, 0i32);
            for _ in 0..freq {
                let pos = io!(postings.next_position());
                if pos < 0 {
                    return err(format!("term {t}: doc {doc}: pos {pos} is out of bounds"));
                }
                if pos > MAX_POSITION {
                    return err(format!(
                        "term {t}: doc {doc}: pos {pos} > IndexWriter.MAX_POSITION={MAX_POSITION}"
                    ));
                }
                if pos < last_pos {
                    return err(format!(
                        "term {t}: doc {doc}: pos {pos} < lastPos {last_pos}"
                    ));
                }
                last_pos = pos;
                if postings.payload().is_some_and(<[u8]>::is_empty) {
                    return err(format!(
                        "term {t}: doc {doc}: pos {pos} payload length is out of bounds 0"
                    ));
                }
                if has_offsets {
                    let (start, end) = (postings.start_offset(), postings.end_offset());
                    if start < 0 {
                        return err(format!(
                            "term {t}: doc {doc}: pos {pos}: startOffset {start} is out of bounds"
                        ));
                    }
                    if start < last_offset {
                        return err(format!(
                            "term {t}: doc {doc}: pos {pos}: startOffset {start} < \
                             lastStartOffset {last_offset}"
                        ));
                    }
                    if end < 0 {
                        return err(format!(
                            "term {t}: doc {doc}: pos {pos}: endOffset {end} is out of bounds"
                        ));
                    }
                    if end < start {
                        return err(format!(
                            "term {t}: doc {doc}: pos {pos}: endOffset {end} < startOffset {start}"
                        ));
                    }
                    last_offset = start;
                }
            }
        }
    }
    Ok((doc_count, total_tf))
}

/// The seven `advance(((idx + 1) * maxDoc) / 8)` probes, each followed by
/// one `nextDoc`; with positions, every probed document's positions too.
fn check_advance(
    te: &mut dyn lucene_search::reader::TermsEnum,
    term: &[u8],
    has_freqs: bool,
    has_positions: bool,
    max_doc: i32,
) -> Check {
    let t = bytes(term);
    let flags = if has_positions {
        PostingsFlags::Positions
    } else {
        PostingsFlags::None
    };
    for i in 1..=7i64 {
        let skip = i32::try_from(i * i64::from(max_doc) / 8).unwrap_or(i32::MAX);
        let mut postings = io!(te.postings(flags));
        let doc = io!(postings.advance(skip));
        if doc == NO_MORE_DOCS {
            break;
        }
        if doc < skip {
            return err(format!(
                "term {t}: advance(docID={skip}) returned docID={doc}"
            ));
        }
        if has_positions {
            let freq = postings.freq();
            if has_freqs && freq <= 0 {
                return err(format!("termFreq {freq} is out of bounds"));
            }
            let mut last = -1i32;
            for _ in 0..freq {
                let pos = io!(postings.next_position());
                if pos < 0 {
                    return err(format!("position {pos} is out of bounds"));
                }
                if pos < last {
                    return err(format!("position {pos} is < lastPosition {last}"));
                }
                last = pos;
            }
        }
        let next = io!(postings.next_doc());
        if next == NO_MORE_DOCS {
            break;
        }
        if next <= doc {
            return err(format!(
                "term {t}: advance(docID={skip}), then .next() returned docID={next} vs \
                 prev docID={doc}"
            ));
        }
    }
    Ok(())
}

/// The slow level's impacts walk: an `ImpactsEnum` in step with a
/// `PostingsEnum`, every block's impacts checked and no frequency above the
/// level-0 bound.
fn check_term_impacts(te: &mut dyn lucene_search::reader::TermsEnum, term: &[u8]) -> Check {
    let t = bytes(term);
    let mut impacts = io!(te.impacts(PostingsFlags::Freqs));
    let mut postings = io!(te.postings(PostingsFlags::Freqs));
    let (mut max, mut max_freq) = (-1i32, 0i32);
    loop {
        let doc = io!(impacts.next_doc());
        let pdoc = io!(postings.next_doc());
        if pdoc != doc {
            return err(format!(
                "Wrong next doc: {doc}, expected {pdoc} for term {t}"
            ));
        }
        if doc == NO_MORE_DOCS {
            break;
        }
        if postings.freq() != impacts.freq() {
            return err(format!(
                "Wrong freq: {}, expected {} for term {t}",
                impacts.freq(),
                postings.freq()
            ));
        }
        if doc > max {
            io!(impacts.advance_shallow(doc));
            let levels = impacts.impacts();
            check_impacts(&levels, doc)?;
            let Some((up_to, level0)) = levels.first() else {
                return err("The number of impact levels must be >= 1, got 0");
            };
            max = *up_to;
            max_freq = level0.last().map_or(0, |i| i.freq);
        }
        if impacts.freq() > max_freq {
            return err(format!(
                "freq {} is greater than the max freq according to impacts {max_freq}",
                impacts.freq()
            ));
        }
    }
    Ok(())
}

/// The norms half of `checkFields`: a live document has a non-zero norm
/// exactly when the field's postings visited it.
fn check_norms_vs_postings(reader: &dyn LeafReader, field: &str, visited: &FixedBitSet) -> Check {
    let Some(mut norms) = io!(reader.norm_values(field)) else {
        return Ok(());
    };
    let mut actual = 0usize;
    loop {
        let doc = io!(norms.next_doc());
        if doc == NO_MORE_DOCS {
            break;
        }
        if !is_live(reader, doc) {
            continue;
        }
        let seen = idx(doc) < visited.len() && visited.get(idx(doc));
        if norms.long_value() != 0 {
            actual = actual.saturating_add(1);
            if !seen {
                return err(format!(
                    "Document {doc} doesn't have terms according to postings but has a norm \
                     value that is not zero: {}",
                    norms.long_value()
                ));
            }
        } else if seen {
            return err(format!(
                "Document {doc} has terms according to postings but its norm value is 0, \
                 which may only be used on documents that have no terms"
            ));
        }
    }
    let mut expected = 0usize;
    visited.for_each_set_bit(|doc| {
        if reader
            .live_docs()
            .is_none_or(|bits| doc < bits.len() && bits.get(doc))
        {
            expected = expected.saturating_add(1);
        }
    });
    if expected != actual {
        return err(format!(
            "Wrong number of documents with norms: expected {expected} but got {actual}"
        ));
    }
    Ok(())
}

/// Seeking back to the field's last term finds it with the same document
/// count, and every term seeks exactly with `seekExact`.
fn check_last_term(te: &mut dyn lucene_search::reader::TermsEnum, last: &[u8]) -> Check {
    if io!(te.try_seek_ceil(last)) != SeekStatus::Found {
        return err(format!("seek to last term {} failed", bytes(last)));
    }
    if te.term() != Some(last) {
        return err(format!(
            "seek to last term {} returned FOUND but seeked to the wrong term {}",
            bytes(last),
            te.term().map(bytes).unwrap_or_default()
        ));
    }
    let expected = io!(te.doc_freq());
    let mut postings = io!(te.postings(PostingsFlags::None));
    let mut doc_freq = 0i32;
    while io!(postings.next_doc()) != NO_MORE_DOCS {
        doc_freq = doc_freq.saturating_add(1);
    }
    if doc_freq != expected {
        return err(format!(
            "docFreq for last term {}={expected} != recomputed docFreq={doc_freq}",
            bytes(last)
        ));
    }
    Ok(())
}

/// `testStoredFields`: every live document's stored fields decode, and each
/// visited field number is a field of the leaf.
fn test_stored_fields(reader: &dyn LeafReader) -> Check {
    struct Visitor<'a> {
        infos: &'a lucene_codecs::field_infos::FieldInfos,
        bad: Option<i32>,
    }
    impl StoredFieldVisitor for Visitor<'_> {
        fn needs_field(
            &mut self,
            number: i32,
        ) -> lucene_codecs::stored_fields::Result<VisitStatus> {
            if self.infos.field_by_number(number).is_none() {
                self.bad.get_or_insert(number);
            }
            Ok(VisitStatus::Yes)
        }
    }
    for doc in 0..reader.max_doc() {
        if !is_live(reader, doc) {
            continue;
        }
        let mut v = Visitor {
            infos: reader.field_infos(),
            bad: None,
        };
        io!(reader.document(doc, &mut v));
        if let Some(number) = v.bad {
            return err(format!(
                "doc {doc}: stored field number {number} is not in the field infos"
            ));
        }
    }
    Ok(())
}

/// `testTermVectors`: each live document's vectors are fields that store
/// them, with ascending terms and positions; at the slow level each vector
/// term's postings in the index hold the document with the same frequency,
/// positions and offsets.
fn test_term_vectors(reader: &dyn LeafReader, slow: bool) -> Check {
    for doc in 0..reader.max_doc() {
        if !is_live(reader, doc) {
            continue;
        }
        let Some(tv) = io!(reader.term_vectors(doc)) else {
            continue;
        };
        for field in &tv.fields {
            let Some(fi) = reader.field_infos().field_by_number(field.field_number) else {
                return err(format!(
                    "docID={doc} has term vectors for field number {} which is not in the \
                     field infos",
                    field.field_number
                ));
            };
            if !fi.store_term_vectors {
                return err(format!(
                    "docID={doc} has term vectors for field={} but FieldInfo has \
                     storeTermVector=false",
                    fi.name
                ));
            }
            let mut last: Option<&[u8]> = None;
            for vt in &field.terms {
                if last.is_some_and(|l| l >= vt.term.as_slice()) {
                    return err(format!(
                        "terms out of order: lastTerm={} term={}",
                        last.map(bytes).unwrap_or_default(),
                        bytes(&vt.term)
                    ));
                }
                last = Some(&vt.term);
                if vt.freq <= 0 {
                    return err(format!("termFreq {} is out of bounds", vt.freq));
                }
                if let Some(positions) = &vt.positions {
                    if positions.windows(2).any(|w| w[1] < w[0]) || positions.iter().any(|&p| p < 0)
                    {
                        return err(format!(
                            "docID={doc} field={} term={}: positions out of order",
                            fi.name,
                            bytes(&vt.term)
                        ));
                    }
                }
                if slow {
                    cross_check_vector_term(reader, doc, fi, vt)?;
                }
            }
        }
    }
    Ok(())
}

fn cross_check_vector_term(
    reader: &dyn LeafReader,
    doc: i32,
    fi: &FieldInfo,
    vt: &lucene_codecs::term_vectors::TermVectorTerm,
) -> Check {
    if fi.index_options == IndexOptions::None {
        return Ok(());
    }
    let t = bytes(&vt.term);
    let field = fi.name.as_str();
    let Some(mut postings) = io!(reader.postings(field, &vt.term, PostingsFlags::All)) else {
        return err(format!(
            "vector term={t} field={field} does not exist in postings; doc={doc}"
        ));
    };
    let got = io!(postings.advance(doc));
    if got != doc {
        return err(format!(
            "vector term={t} field={field}: doc={doc} was not found in postings (got: {got})"
        ));
    }
    if !has_freqs(fi.index_options) {
        return Ok(());
    }
    if postings.freq() != vt.freq {
        return err(format!(
            "vector term={t} field={field} doc={doc}: freq={} differs from postings freq={}",
            vt.freq,
            postings.freq()
        ));
    }
    if !fi.index_options.subsumes_positions() {
        return Ok(());
    }
    for i in 0..idx(vt.freq) {
        let pos = io!(postings.next_position());
        if let Some(&vpos) = vt.positions.as_ref().and_then(|p| p.get(i)) {
            if vpos != pos {
                return err(format!(
                    "vector term={t} field={field} doc={doc}: pos={vpos} differs from \
                     postings pos={pos}"
                ));
            }
        }
        if fi.index_options.subsumes_offsets() {
            let start = vt.start_offsets.as_ref().and_then(|o| o.get(i));
            let end = vt.end_offsets.as_ref().and_then(|o| o.get(i));
            if let (Some(&s), Some(&e)) = (start, end) {
                if s != postings.start_offset() || e != postings.end_offset() {
                    return err(format!(
                        "vector term={t} field={field} doc={doc}: offsets {s}-{e} differ from \
                         postings offsets {}-{}",
                        postings.start_offset(),
                        postings.end_offset()
                    ));
                }
            }
        }
    }
    Ok(())
}

/// `testDocValues` -> `checkDocValues`: each field's iterator against a
/// second one positioned by `advanceExact`.
fn test_doc_values(reader: &dyn LeafReader) -> Check {
    for fi in &reader.field_infos().fields {
        let f = fi.name.as_str();
        match fi.doc_values_type {
            DocValuesType::None => {}
            DocValuesType::Numeric => {
                let (Some(mut a), Some(mut b)) = (
                    io!(reader.numeric_doc_values(f)),
                    io!(reader.numeric_doc_values(f)),
                ) else {
                    return err(format!(
                        "field: {f} has numeric docvalues but reader has none"
                    ));
                };
                check_unpositioned(f, a.doc_id())?;
                loop {
                    let doc = io!(a.next_doc());
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    if !io!(b.advance_exact(doc)) {
                        return err(format!("advanceExact did not find matching doc ID: {doc}"));
                    }
                    if a.long_value() != b.long_value() {
                        return err(format!(
                            "nextDoc and advanceExact report different values: {} != {}",
                            a.long_value(),
                            b.long_value()
                        ));
                    }
                }
                check_bulk_numeric(reader, f)?;
            }
            DocValuesType::Binary => {
                let (Some(mut a), Some(mut b)) = (
                    io!(reader.binary_doc_values(f)),
                    io!(reader.binary_doc_values(f)),
                ) else {
                    return err(format!(
                        "field: {f} has binary docvalues but reader has none"
                    ));
                };
                check_unpositioned(f, a.doc_id())?;
                loop {
                    let doc = io!(a.next_doc());
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    if !io!(b.advance_exact(doc)) {
                        return err(format!("advanceExact did not find matching doc ID: {doc}"));
                    }
                    if a.binary_value() != b.binary_value() {
                        return err(format!(
                            "nextDoc and advanceExact report different values: {} != {}",
                            bytes(a.binary_value()),
                            bytes(b.binary_value())
                        ));
                    }
                }
            }
            DocValuesType::Sorted => check_sorted(reader, f)?,
            DocValuesType::SortedSet => check_sorted_set(reader, f)?,
            DocValuesType::SortedNumeric => check_sorted_numeric(reader, f)?,
        }
    }
    Ok(())
}

fn check_unpositioned(field: &str, doc: i32) -> Check {
    if doc != -1 {
        return err(format!(
            "dv iterator for field: {field} should start at docID=-1, but got {doc}"
        ));
    }
    Ok(())
}

/// `checkBulkFetchNumericDocValues`: `longValues` over runs of 16 documents
/// stepping 1-4 apart, against `advanceExact`.
fn check_bulk_numeric(reader: &dyn LeafReader, f: &str) -> Check {
    let (Some(mut a), Some(mut b)) = (
        io!(reader.numeric_doc_values(f)),
        io!(reader.numeric_doc_values(f)),
    ) else {
        return Ok(());
    };
    let max_doc = reader.max_doc();
    let mut doc = -1i32;
    while doc < max_doc {
        let mut docs = Vec::with_capacity(16);
        for j in 0..16i32 {
            doc = doc.saturating_add(1 + (j & 3));
            if doc >= max_doc {
                break;
            }
            docs.push(doc);
        }
        let mut values = vec![0i64; docs.len()];
        io!(a.long_values(&docs, &mut values, 42));
        for (&d, &v) in docs.iter().zip(&values) {
            let expected = if io!(b.advance_exact(d)) {
                b.long_value()
            } else {
                42
            };
            if v != expected {
                return err(format!(
                    "Bulk retrieval of numeric doc values for field {f} at doc {d} got {v} but \
                     expected {expected}"
                ));
            }
        }
    }
    Ok(())
}

fn check_sorted(reader: &dyn LeafReader, f: &str) -> Check {
    let (Some(mut a), Some(mut b)) = (
        io!(reader.sorted_doc_values(f)),
        io!(reader.sorted_doc_values(f)),
    ) else {
        return err(format!(
            "field: {f} has sorted docvalues but reader has none"
        ));
    };
    check_unpositioned(f, a.doc_id())?;
    let max_ord = a.value_count().saturating_sub(1);
    let mut seen = FixedBitSet::new(idx(a.value_count()));
    let mut max_ord2 = -1i32;
    loop {
        let doc = io!(a.next_doc());
        if doc == NO_MORE_DOCS {
            break;
        }
        let ord = a.ord_value();
        if ord == -1 {
            return err(format!("dv for field: {f} has -1 ord"));
        }
        if ord < -1 || ord > max_ord {
            return err(format!("ord out of bounds: {ord}"));
        }
        max_ord2 = max_ord2.max(ord);
        // FBS: `seen` holds `valueCount` bits and `ord` was checked to be in `0..=maxOrd`.
        seen.set(idx(ord));
        if !io!(b.advance_exact(doc)) {
            return err(format!("advanceExact did not find matching doc ID: {doc}"));
        }
        if ord != b.ord_value() {
            return err(format!(
                "nextDoc and advanceExact report different ords: {ord} != {}",
                b.ord_value()
            ));
        }
    }
    if max_ord != max_ord2 {
        return err(format!(
            "dv for field: {f} reports wrong maxOrd={max_ord} but this is not the case: {max_ord2}"
        ));
    }
    if i32::try_from(seen.cardinality()).ok() != Some(a.value_count()) {
        return err(format!(
            "dv for field: {f} has holes in its ords, valueCount={} but only used: {}",
            a.value_count(),
            seen.cardinality()
        ));
    }
    let mut last: Option<Vec<u8>> = None;
    for ord in 0..=max_ord {
        let term = io!(a.lookup_ord(ord));
        if last.as_ref().is_some_and(|l| *l >= term) {
            return err(format!(
                "dv for field: {f} has ords out of order: {} >={}",
                last.as_deref().map(bytes).unwrap_or_default(),
                bytes(&term)
            ));
        }
        last = Some(term);
    }
    Ok(())
}

fn check_sorted_set(reader: &dyn LeafReader, f: &str) -> Check {
    let (Some(mut a), Some(mut b)) = (
        io!(reader.sorted_set_doc_values(f)),
        io!(reader.sorted_set_doc_values(f)),
    ) else {
        return err(format!(
            "field: {f} has sorted-set docvalues but reader has none"
        ));
    };
    let max_ord = a.value_count().saturating_sub(1);
    let mut seen = FixedBitSet::new(usize::try_from(a.value_count()).unwrap_or(0));
    let mut max_ord2 = -1i64;
    loop {
        let doc = io!(a.next_doc());
        if doc == NO_MORE_DOCS {
            break;
        }
        let count = a.doc_value_count();
        if count == 0 {
            return err(format!(
                "sortedset dv for field: {f} returned docValueCount=0 for docID={doc}"
            ));
        }
        if !io!(b.advance_exact(doc)) {
            return err(format!("advanceExact did not find matching doc ID: {doc}"));
        }
        if count != b.doc_value_count() {
            return err(format!(
                "advanceExact reports different value count: {count} != {}",
                b.doc_value_count()
            ));
        }
        let mut last_ord = -1i64;
        for _ in 0..count {
            let ord = io!(a.next_ord());
            let ord2 = io!(b.next_ord());
            if ord != ord2 {
                return err(format!(
                    "nextDoc and advanceExact report different ords: {ord} != {ord2}"
                ));
            }
            if ord <= last_ord {
                return err(format!(
                    "ords out of order: {ord} <= {last_ord} for doc: {doc}"
                ));
            }
            if ord < 0 || ord > max_ord {
                return err(format!("ord out of bounds: {ord}"));
            }
            last_ord = ord;
            max_ord2 = max_ord2.max(ord);
            // FBS: `seen` holds `valueCount` bits and `ord` was checked to be in `0..=maxOrd`.
            seen.set(usize::try_from(ord).unwrap_or(0));
        }
    }
    if max_ord != max_ord2 {
        return err(format!(
            "dv for field: {f} reports wrong maxOrd={max_ord} but this is not the case: {max_ord2}"
        ));
    }
    if i64::try_from(seen.cardinality()).ok() != Some(a.value_count()) {
        return err(format!(
            "dv for field: {f} has holes in its ords, valueCount={} but only used: {}",
            a.value_count(),
            seen.cardinality()
        ));
    }
    let mut last: Option<Vec<u8>> = None;
    for ord in 0..=max_ord {
        let term = io!(a.lookup_ord(ord));
        if last.as_ref().is_some_and(|l| *l >= term) {
            return err(format!(
                "dv for field: {f} has ords out of order: {} >={}",
                last.as_deref().map(bytes).unwrap_or_default(),
                bytes(&term)
            ));
        }
        last = Some(term);
    }
    Ok(())
}

fn check_sorted_numeric(reader: &dyn LeafReader, f: &str) -> Check {
    let (Some(mut a), Some(mut b)) = (
        io!(reader.sorted_numeric_doc_values(f)),
        io!(reader.sorted_numeric_doc_values(f)),
    ) else {
        return err(format!(
            "field: {f} has sorted-numeric docvalues but reader has none"
        ));
    };
    check_unpositioned(f, a.doc_id())?;
    loop {
        let doc = io!(a.next_doc());
        if doc == NO_MORE_DOCS {
            break;
        }
        let count = a.doc_value_count();
        if count == 0 {
            return err(format!(
                "sorted numeric dv for field: {f} returned docValueCount=0 for docID={doc}"
            ));
        }
        if !io!(b.advance_exact(doc)) {
            return err(format!("advanceExact did not find matching doc ID: {doc}"));
        }
        if count != b.doc_value_count() {
            return err(format!(
                "advanceExact reports different value count: {count} != {}",
                b.doc_value_count()
            ));
        }
        let mut previous = i64::MIN;
        for _ in 0..count {
            let value = io!(a.next_value());
            if value < previous {
                return err(format!(
                    "values out of order: {value} < {previous} for doc: {doc}"
                ));
            }
            previous = value;
            let value2 = io!(b.next_value());
            if value != value2 {
                return err(format!(
                    "advanceExact reports different value: {value} != {value2}"
                ));
            }
        }
    }
    Ok(())
}

/// The point walk: every point lies in its leaf's bounds and the field's,
/// with the right packed length, a document below `maxDoc`, one-dimension
/// values ascending; the count is `size()` and the distinct documents
/// `getDocCount()`.
struct VerifyPoints<'a> {
    num_dims: usize,
    num_index_dims: usize,
    bytes_per_dim: usize,
    min: &'a [u8],
    max: &'a [u8],
    max_doc: i32,
    docs: FixedBitSet,
    count: i64,
    last_value: Option<Vec<u8>>,
    error: Option<String>,
}

impl VerifyPoints<'_> {
    fn fail(&mut self, msg: String) {
        self.error.get_or_insert(msg);
    }

    fn dim(&self, packed: &'_ [u8], d: usize) -> Option<std::ops::Range<usize>> {
        let start = d.checked_mul(self.bytes_per_dim)?;
        let end = start.checked_add(self.bytes_per_dim)?;
        (end <= packed.len()).then_some(start..end)
    }
}

impl IntersectVisitor for VerifyPoints<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        for d in 0..self.num_index_dims {
            if let Some(r) = self.dim(min_packed, d) {
                if min_packed.get(r.clone()) > max_packed.get(r.clone()) {
                    self.fail(format!(
                        "packed points cell minPackedValue > maxPackedValue at dim={d}"
                    ));
                }
            }
        }
        Relation::CellCrossesQuery
    }

    fn visit(&mut self, doc_id: i32) {
        self.fail(format!(
            "visit(docID={doc_id}) without a value: the walk said CELL_CROSSES_QUERY"
        ));
    }

    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        self.count = self.count.saturating_add(1);
        if doc_id < 0 || doc_id >= self.max_doc {
            self.fail(format!(
                "docID={doc_id} is out of bounds (maxDoc={})",
                self.max_doc
            ));
            return;
        }
        // FBS: `docs` holds `max_doc` bits and `doc_id` was just bounded by it.
        self.docs.set(idx(doc_id));
        let want = self.num_dims.saturating_mul(self.bytes_per_dim);
        if packed.len() != want {
            self.fail(format!(
                "packedValue has length={} but expected {want}",
                packed.len()
            ));
            return;
        }
        for d in 0..self.num_index_dims {
            let Some(r) = self.dim(packed, d) else {
                continue;
            };
            if packed.get(r.clone()) < self.min.get(r.clone()) {
                self.fail(format!(
                    "docID={doc_id} packedValue is out-of-bounds of the field's minPackedValue at dim={d}"
                ));
            }
            if packed.get(r.clone()) > self.max.get(r) {
                self.fail(format!(
                    "docID={doc_id} packedValue is out-of-bounds of the field's maxPackedValue at dim={d}"
                ));
            }
        }
        if self.num_dims == 1 {
            if self.last_value.as_deref().is_some_and(|l| l > packed) {
                self.fail(format!(
                    "packed points value is out of order for 1D field (docID={doc_id})"
                ));
            }
            self.last_value = Some(packed.to_vec());
        }
    }
}

/// `testPoints`.
fn test_points(reader: &dyn LeafReader) -> Check {
    for fi in &reader.field_infos().fields {
        if fi.point_dimension_count == 0 {
            continue;
        }
        let Some(values) = io!(reader.point_values(&fi.name)) else {
            continue;
        };
        let field = &fi.name;
        if values.num_dimensions() != fi.point_dimension_count
            || values.num_index_dimensions() != fi.point_index_dimension_count
            || values.bytes_per_dimension() != fi.point_num_bytes
        {
            return err(format!(
                "field={field}: point values report {}/{}/{} dimensions/index dimensions/bytes \
                 but FieldInfo says {}/{}/{}",
                values.num_dimensions(),
                values.num_index_dimensions(),
                values.bytes_per_dimension(),
                fi.point_dimension_count,
                fi.point_index_dimension_count,
                fi.point_num_bytes
            ));
        }
        let to_usize = |v: i32| usize::try_from(v).unwrap_or(0);
        let mut v = VerifyPoints {
            num_dims: to_usize(values.num_dimensions()),
            num_index_dims: to_usize(values.num_index_dimensions()),
            bytes_per_dim: to_usize(values.bytes_per_dimension()),
            min: values.min_packed_value(),
            max: values.max_packed_value(),
            max_doc: reader.max_doc(),
            docs: max_doc_bits(reader),
            count: 0,
            last_value: None,
            error: None,
        };
        io!(values.intersect(&mut v));
        if let Some(e) = v.error {
            return err(format!("field={field}: {e}"));
        }
        if v.count != values.size() {
            return err(format!(
                "point values for field \"{field}\" claims to have size={} points, but in fact \
                 has {}",
                values.size(),
                v.count
            ));
        }
        let doc_count = v.docs.cardinality();
        if i32::try_from(doc_count).ok() != Some(values.doc_count()) {
            return err(format!(
                "point values for field \"{field}\" claims to have docCount={} but in fact has \
                 {doc_count}",
                values.doc_count()
            ));
        }
    }
    Ok(())
}

/// `testVectors`: each vector field's values have the field's dimension,
/// ordinals map to ascending documents below `maxDoc`, and each vector has
/// that many components.
fn test_vectors(reader: &dyn LeafReader) -> Check {
    use lucene_codecs::field_infos::VectorEncoding;
    for fi in &reader.field_infos().fields {
        if fi.vector_dimension == 0 {
            continue;
        }
        let field = &fi.name;
        let dim = usize::try_from(fi.vector_dimension).unwrap_or(0);
        let max_doc = reader.max_doc();
        let check_docs =
            |size: i32, ord_to_doc: &dyn Fn(i32) -> lucene_search::Result<i32>| -> Check {
                let mut last = -1i32;
                for ord in 0..size {
                    let doc = io!(ord_to_doc(ord));
                    if doc <= last || doc >= max_doc {
                        return err(format!(
                            "field={field}: ord {ord} maps to docID={doc} after docID={last} \
                         (maxDoc={max_doc})"
                        ));
                    }
                    last = doc;
                }
                Ok(())
            };
        match fi.vector_encoding {
            VectorEncoding::Float32 => {
                let Some(values) = io!(reader.float_vector_values(field)) else {
                    continue;
                };
                if values.dimension() != dim {
                    return err(format!(
                        "Field \"{field}\" has vector values of dimension {} but FieldInfo says {dim}",
                        values.dimension()
                    ));
                }
                check_docs(values.size(), &|o| values.ord_to_doc(o))?;
                for ord in 0..values.size() {
                    if io!(values.vector_value(ord)).len() != dim {
                        return err(format!(
                            "Field \"{field}\": ord {ord} has the wrong dimension"
                        ));
                    }
                }
            }
            VectorEncoding::Byte => {
                let Some(values) = io!(reader.byte_vector_values(field)) else {
                    continue;
                };
                if values.dimension() != dim {
                    return err(format!(
                        "Field \"{field}\" has vector values of dimension {} but FieldInfo says {dim}",
                        values.dimension()
                    ));
                }
                check_docs(values.size(), &|o| values.ord_to_doc(o))?;
                for ord in 0..values.size() {
                    if io!(values.vector_value(ord)).len() != dim {
                        return err(format!(
                            "Field \"{field}\": ord {ord} has the wrong dimension"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// `checkReaderSanity`: a field without norms gives none, and a field's doc
/// values come in its one type only.
fn check_reader_sanity(reader: &dyn LeafReader) -> Check {
    for fi in &reader.field_infos().fields {
        let f = fi.name.as_str();
        let has_norms = fi.index_options != IndexOptions::None && !fi.omit_norms;
        if !has_norms && io!(reader.norm_values(f)).is_some() {
            return err(format!("field: {f} should omit norms but has them!"));
        }
        let present = [
            (
                DocValuesType::Numeric,
                io!(reader.numeric_doc_values(f)).is_some(),
            ),
            (
                DocValuesType::Binary,
                io!(reader.binary_doc_values(f)).is_some(),
            ),
            (
                DocValuesType::Sorted,
                io!(reader.sorted_doc_values(f)).is_some(),
            ),
            (
                DocValuesType::SortedNumeric,
                io!(reader.sorted_numeric_doc_values(f)).is_some(),
            ),
            (
                DocValuesType::SortedSet,
                io!(reader.sorted_set_doc_values(f)).is_some(),
            ),
        ];
        let other = present
            .iter()
            .any(|&(t, has)| has && t != fi.doc_values_type);
        if other {
            return err(if fi.doc_values_type == DocValuesType::None {
                format!("field: {f} has docvalues but should omit them!")
            } else {
                format!("{f} returns multiple docvalues types!")
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    type Walk = dyn Fn(&mut VerifyPoints<'_>);

    fn verifier<'a>(min: &'a [u8], max: &'a [u8], dims: usize) -> VerifyPoints<'a> {
        VerifyPoints {
            num_dims: dims,
            num_index_dims: dims,
            bytes_per_dim: 1,
            min,
            max,
            max_doc: 4,
            docs: FixedBitSet::new(4),
            count: 0,
            last_value: None,
            error: None,
        }
    }

    #[test]
    fn the_point_verifier_reports_the_first_bad_point() {
        let cases: [(&Walk, &str); 7] = [
            (
                &|v| {
                    let _ = v.compare(&[5], &[4]);
                },
                "minPackedValue > maxPackedValue",
            ),
            (&|v| v.visit(1), "without a value"),
            (&|v| v.visit_with_value(9, &[3]), "out of bounds"),
            (&|v| v.visit_with_value(1, &[3, 3]), "has length=2"),
            (&|v| v.visit_with_value(1, &[1]), "minPackedValue at dim=0"),
            (&|v| v.visit_with_value(1, &[9]), "maxPackedValue at dim=0"),
            (
                &|v| {
                    v.visit_with_value(1, &[5]);
                    v.visit_with_value(2, &[3]);
                },
                "out of order for 1D",
            ),
        ];
        for (walk, message) in cases {
            let mut v = verifier(&[2], &[6], 1);
            walk(&mut v);
            let e = v.error.unwrap_or_default();
            assert!(e.contains(message), "{message}: {e}");
        }
        let mut v = verifier(&[2], &[6], 1);
        assert_eq!(v.compare(&[2], &[6]), Relation::CellCrossesQuery);
        v.visit_with_value(0, &[2]);
        v.visit_with_value(3, &[6]);
        assert!(v.error.is_none());
        assert_eq!((v.count, v.docs.cardinality()), (2, 2));
    }

    #[test]
    fn helpers() {
        assert!(check_unpositioned("f", 0).unwrap_err().contains("docID=-1"));
        assert!(check_unpositioned("f", -1).is_ok());
        assert_eq!(bytes(b"ab"), "[61 62]");
        assert!(has_freqs(IndexOptions::DocsAndFreqs));
        assert!(!has_freqs(IndexOptions::Docs));
    }
}
