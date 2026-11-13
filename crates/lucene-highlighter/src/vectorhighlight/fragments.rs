//! `FragmentsBuilder` (`BaseFragmentsBuilder` with `SimpleFragmentsBuilder`
//! and `ScoreOrderFragmentsBuilder`) and the `BoundaryScanner`s: the text of
//! each fragment cut from the stored field values, with its phrases marked.
//!
//! Text is UTF-16, as Java's `StringBuilder` indexes it, so the offsets of
//! the term vector address it directly.

use lucene_codecs::stored_fields::{DocumentVisitor, FieldValue};

use super::super::unified::break_iterators::BreakIterator;
use super::super::{DefaultEncoder, Encoder, Error, Result};
use super::field_phrase_list::Toffs;
use super::frag_list::{FieldFragList, SubInfo, WeightedFragInfo};
use lucene_search::directory_reader::DirectoryReader;

/// `BaseFragmentsBuilder.COLORED_PRE_TAGS`.
pub const COLORED_PRE_TAGS: [&str; 20] = [
    "<b style=\"background:yellow\">",
    "<b style=\"background:lawngreen\">",
    "<b style=\"background:aquamarine\">",
    "<b style=\"background:magenta\">",
    "<b style=\"background:palegreen\">",
    "<b style=\"background:coral\">",
    "<b style=\"background:wheat\">",
    "<b style=\"background:khaki\">",
    "<b style=\"background:lime\">",
    "<b style=\"background:deepskyblue\">",
    "<b style=\"background:deeppink\">",
    "<b style=\"background:salmon\">",
    "<b style=\"background:peachpuff\">",
    "<b style=\"background:violet\">",
    "<b style=\"background:mediumpurple\">",
    "<b style=\"background:palegoldenrod\">",
    "<b style=\"background:darkkhaki\">",
    "<b style=\"background:springgreen\">",
    "<b style=\"background:turquoise\">",
    "<b style=\"background:powderblue\">",
];

/// `BaseFragmentsBuilder.COLORED_POST_TAGS`.
pub const COLORED_POST_TAGS: [&str; 1] = ["</b>"];

/// `BoundaryScanner`: where a fragment may start and end.
pub trait BoundaryScanner {
    /// `findStartOffset(buffer, start)`.
    fn find_start_offset(&mut self, buffer: &[u16], start: i32) -> i32;
    /// `findEndOffset(buffer, start)`.
    fn find_end_offset(&mut self, buffer: &[u16], start: i32) -> i32;
}

fn len_i32(buffer: &[u16]) -> i32 {
    i32::try_from(buffer.len()).unwrap_or(i32::MAX)
}

/// `SimpleBoundaryScanner`: up to `maxScan` characters to the nearest
/// boundary character.
#[derive(Debug, Clone)]
pub struct SimpleBoundaryScanner {
    max_scan: i32,
    boundary_chars: Vec<u16>,
}

impl SimpleBoundaryScanner {
    /// `DEFAULT_MAX_SCAN`.
    pub const DEFAULT_MAX_SCAN: i32 = 20;
    /// `DEFAULT_BOUNDARY_CHARS`.
    pub const DEFAULT_BOUNDARY_CHARS: [u16; 7] = [46, 44, 33, 63, 32, 9, 10];

    /// `new SimpleBoundaryScanner()`.
    pub fn new() -> Self {
        Self::with_max_scan(Self::DEFAULT_MAX_SCAN)
    }

    /// `new SimpleBoundaryScanner(maxScan)`.
    pub fn with_max_scan(max_scan: i32) -> Self {
        Self::with_chars(max_scan, &Self::DEFAULT_BOUNDARY_CHARS)
    }

    /// `new SimpleBoundaryScanner(maxScan, boundaryChars)`.
    pub fn with_chars(max_scan: i32, boundary_chars: &[u16]) -> Self {
        Self {
            max_scan,
            boundary_chars: boundary_chars.to_vec(),
        }
    }
}

impl Default for SimpleBoundaryScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundaryScanner for SimpleBoundaryScanner {
    fn find_start_offset(&mut self, buffer: &[u16], start: i32) -> i32 {
        if start > len_i32(buffer) || start < 1 {
            return start;
        }
        let (mut offset, mut count) = (start, self.max_scan);
        while offset > 0 && count > 0 {
            if self.boundary_chars.contains(&buffer[(offset - 1) as usize]) {
                return offset;
            }
            offset -= 1;
            count -= 1;
        }
        if offset == 0 {
            return 0;
        }
        start
    }

    fn find_end_offset(&mut self, buffer: &[u16], start: i32) -> i32 {
        if start > len_i32(buffer) || start < 0 {
            return start;
        }
        let (mut offset, mut count) = (start, self.max_scan);
        while offset < len_i32(buffer) && count > 0 {
            if self.boundary_chars.contains(&buffer[offset as usize]) {
                return offset;
            }
            offset += 1;
            count -= 1;
        }
        start
    }
}

/// `BreakIteratorBoundaryScanner`: the boundaries of a break iterator.
pub struct BreakIteratorBoundaryScanner {
    bi: Box<dyn BreakIterator>,
}

impl BreakIteratorBoundaryScanner {
    /// `new BreakIteratorBoundaryScanner(bi)`.
    pub fn new(bi: Box<dyn BreakIterator>) -> Self {
        Self { bi }
    }
}

impl BoundaryScanner for BreakIteratorBoundaryScanner {
    fn find_start_offset(&mut self, buffer: &[u16], start: i32) -> i32 {
        if start > len_i32(buffer) || start < 1 {
            return start;
        }
        self.bi.set_text(&buffer[..start as usize]);
        self.bi.last();
        self.bi.previous()
    }

    fn find_end_offset(&mut self, buffer: &[u16], start: i32) -> i32 {
        if start > len_i32(buffer) || start < 0 {
            return start;
        }
        self.bi.set_text(&buffer[start as usize..]);
        self.bi.next().wrapping_add(start)
    }
}

/// The order a [`FragmentsBuilder`] returns its fragments in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentsOrder {
    /// `SimpleFragmentsBuilder`: as they come, by position.
    Simple,
    /// `ScoreOrderFragmentsBuilder`: the best first.
    ScoreOrder,
}

/// `BaseFragmentsBuilder`.
pub struct FragmentsBuilder {
    order: FragmentsOrder,
    pre_tags: Vec<String>,
    post_tags: Vec<String>,
    multi_valued_separator: u16,
    boundary_scanner: Box<dyn BoundaryScanner>,
    discrete_multi_value_highlighting: bool,
    fix_broken_analysis: bool,
    strict_utf16: bool,
}

impl FragmentsBuilder {
    /// A builder in `order` with the default tags and boundary scanner.
    pub fn new(order: FragmentsOrder) -> Self {
        Self {
            order,
            pre_tags: vec!["<b>".to_string()],
            post_tags: vec!["</b>".to_string()],
            multi_valued_separator: u16::from(b' '),
            boundary_scanner: Box::new(SimpleBoundaryScanner::new()),
            discrete_multi_value_highlighting: false,
            fix_broken_analysis: false,
            strict_utf16: false,
        }
    }

    /// `SimpleFragmentsBuilder()`.
    pub fn simple() -> Self {
        Self::new(FragmentsOrder::Simple)
    }

    /// `ScoreOrderFragmentsBuilder()`.
    pub fn score_order() -> Self {
        Self::new(FragmentsOrder::ScoreOrder)
    }

    /// The tags `createFragments` marks with when it is given none.
    #[must_use]
    pub fn with_tags(mut self, pre_tags: &[&str], post_tags: &[&str]) -> Self {
        self.pre_tags = pre_tags.iter().map(|s| (*s).to_string()).collect();
        self.post_tags = post_tags.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// The boundary scanner.
    #[must_use]
    pub fn with_boundary_scanner(mut self, scanner: Box<dyn BoundaryScanner>) -> Self {
        self.boundary_scanner = scanner;
        self
    }

    /// `setMultiValuedSeparator(separator)`.
    pub fn set_multi_valued_separator(&mut self, separator: u16) {
        self.multi_valued_separator = separator;
    }

    /// `getMultiValuedSeparator()`.
    pub fn multi_valued_separator(&self) -> u16 {
        self.multi_valued_separator
    }

    /// `setDiscreteMultiValueHighlighting(discrete)`.
    pub fn set_discrete_multi_value_highlighting(&mut self, discrete: bool) {
        self.discrete_multi_value_highlighting = discrete;
    }

    /// OpenSearch's `SimpleFragmentsBuilder` over a field whose analyzer has
    /// a filter that `breaksFastVectorHighlighter`: each fragment starts no
    /// later than its first match (`FragmentBuilderHelper.fixWeightedFragInfo`).
    pub fn set_fix_broken_analysis(&mut self, fix: bool) {
        self.fix_broken_analysis = fix;
    }

    /// A cut through a surrogate pair, or an unpaired surrogate in the text,
    /// is an error instead of U+FFFD, for a caller that must give Java's
    /// UTF-16 exactly.
    pub fn set_strict_utf16(&mut self, strict: bool) {
        self.strict_utf16 = strict;
    }

    /// `isDiscreteMultiValueHighlighting()`.
    pub fn is_discrete_multi_value_highlighting(&self) -> bool {
        self.discrete_multi_value_highlighting
    }

    /// `createFragment(reader, docId, fieldName, fieldFragList)`.
    ///
    /// # Errors
    /// As [`Self::create_fragments`].
    pub fn create_fragment(
        &mut self,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_list: &FieldFragList,
    ) -> Result<Option<String>> {
        let (pre, post) = (self.pre_tags.clone(), self.post_tags.clone());
        self.create_fragment_with(reader, doc, field, frag_list, &pre, &post, &DefaultEncoder)
    }

    /// `createFragments(reader, docId, fieldName, fieldFragList,
    /// maxNumFragments)`.
    ///
    /// # Errors
    /// As [`Self::create_fragments_with`].
    pub fn create_fragments(
        &mut self,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_list: &FieldFragList,
        max_num_fragments: i32,
    ) -> Result<Option<Vec<String>>> {
        let (pre, post) = (self.pre_tags.clone(), self.post_tags.clone());
        self.create_fragments_with(
            reader,
            doc,
            field,
            frag_list,
            max_num_fragments,
            &pre,
            &post,
            &DefaultEncoder,
        )
    }

    /// `createFragment(..., preTags, postTags, encoder)`: the first fragment.
    ///
    /// # Errors
    /// As [`Self::create_fragments_with`].
    #[allow(clippy::too_many_arguments)]
    pub fn create_fragment_with(
        &mut self,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_list: &FieldFragList,
        pre_tags: &[String],
        post_tags: &[String],
        encoder: &dyn Encoder,
    ) -> Result<Option<String>> {
        Ok(self
            .create_fragments_with(
                reader, doc, field, frag_list, 1, pre_tags, post_tags, encoder,
            )?
            .and_then(|f| f.into_iter().next()))
    }

    /// `createFragments(..., maxNumFragments, preTags, postTags, encoder)`:
    /// `None` for a document without a stored value of the field.
    ///
    /// # Errors
    /// A negative `max_num_fragments`, empty tags, a document the reader
    /// does not have, or offsets that lie outside the stored text.
    #[allow(clippy::too_many_arguments)]
    pub fn create_fragments_with(
        &mut self,
        reader: &DirectoryReader,
        doc: i32,
        field: &str,
        frag_list: &FieldFragList,
        max_num_fragments: i32,
        pre_tags: &[String],
        post_tags: &[String],
        encoder: &dyn Encoder,
    ) -> Result<Option<Vec<String>>> {
        if max_num_fragments < 0 {
            return Err(Error::IllegalArgument(format!(
                "maxNumFragments({max_num_fragments}) must be positive number."
            )));
        }
        let values = Self::fields(reader, doc, field)?;
        self.create_fragments_from_values(
            &values,
            frag_list,
            max_num_fragments,
            pre_tags,
            post_tags,
            encoder,
        )
    }

    /// `createFragments(..., maxNumFragments, preTags, postTags, encoder)`
    /// over values already loaded (`getFields`'s, in order): `None` for no
    /// values.
    ///
    /// # Errors
    /// A negative `max_num_fragments`, empty tags, or offsets that lie
    /// outside the text.
    #[allow(clippy::too_many_arguments)]
    pub fn create_fragments_from_values(
        &mut self,
        values: &[Vec<u16>],
        frag_list: &FieldFragList,
        max_num_fragments: i32,
        pre_tags: &[String],
        post_tags: &[String],
        encoder: &dyn Encoder,
    ) -> Result<Option<Vec<String>>> {
        if max_num_fragments < 0 {
            return Err(Error::IllegalArgument(format!(
                "maxNumFragments({max_num_fragments}) must be positive number."
            )));
        }
        let mut frag_infos: Vec<WeightedFragInfo> = frag_list.frag_infos().to_vec();
        if values.is_empty() {
            return Ok(None);
        }
        if self.discrete_multi_value_highlighting && values.len() > 1 {
            frag_infos = Self::discrete_multi_value_highlighting(frag_infos, values)?;
        }
        if self.order == FragmentsOrder::ScoreOrder {
            frag_infos.sort_by(|o1, o2| {
                if o1.total_boost > o2.total_boost {
                    std::cmp::Ordering::Less
                } else if o1.total_boost < o2.total_boost {
                    std::cmp::Ordering::Greater
                } else {
                    o1.start_offset.cmp(&o2.start_offset)
                }
            });
        }
        let limit = usize::try_from(max_num_fragments)
            .unwrap_or(usize::MAX)
            .min(frag_infos.len());
        let mut fragments = Vec::with_capacity(limit);
        let mut buffer: Vec<u16> = Vec::new();
        let mut next_value_index = 0usize;
        for frag_info in &mut frag_infos[..limit] {
            if self.fix_broken_analysis {
                fix_weighted_frag_info(frag_info);
            }
            fragments.push(self.make_fragment(
                &mut buffer,
                &mut next_value_index,
                values,
                frag_info,
                pre_tags,
                post_tags,
                encoder,
            )?);
        }
        Ok(Some(fragments))
    }

    /// `getFields(reader, docId, fieldName)`: the stored string values of
    /// the field, in stored order.
    fn fields(reader: &DirectoryReader, doc: i32, field: &str) -> Result<Vec<Vec<u16>>> {
        let segment = reader
            .segment_readers()
            .iter()
            .find(|s| s.doc_base <= doc && doc - s.doc_base < s.max_doc)
            .ok_or_else(|| Error::IllegalArgument(format!("doc {doc} is in no segment")))?;
        let mut values = Vec::new();
        // Java's visitor answers YES for this field and NO for every other,
        // so the rest of the document (a large `_source`) is never decoded.
        let Some(number) = segment
            .field_infos()
            .field_by_name(field)
            .map(|fi| fi.number)
        else {
            return Ok(values);
        };
        let mut visitor = DocumentVisitor::for_fields(&[number]);
        let stored = segment
            .visit_stored_document(doc - segment.doc_base, &mut visitor)
            .map_err(|e| Error::Io(e.to_string()))?;
        if !stored {
            return Ok(values);
        }
        for f in &visitor.into_document().fields {
            if let FieldValue::String(text) = &f.value {
                values.push(text.encode_utf16().collect());
            }
        }
        Ok(values)
    }

    /// `makeFragment`.
    #[allow(clippy::too_many_arguments)]
    fn make_fragment(
        &mut self,
        buffer: &mut Vec<u16>,
        index: &mut usize,
        values: &[Vec<u16>],
        frag_info: &WeightedFragInfo,
        pre_tags: &[String],
        post_tags: &[String],
        encoder: &dyn Encoder,
    ) -> Result<String> {
        let (src, modified_start) = self.fragment_source(
            buffer,
            index,
            values,
            frag_info.start_offset,
            frag_info.end_offset,
        )?;
        let mut fragment = String::new();
        let mut src_index = 0i32;
        for sub in &frag_info.sub_infos {
            for to in &sub.terms_offsets {
                let start = to.start_offset().wrapping_sub(modified_start);
                let end = to.end_offset().wrapping_sub(modified_start);
                fragment.push_str(&encoder.encode_text(&substring(
                    &src,
                    self.strict_utf16,
                    src_index,
                    start,
                )?));
                fragment.push_str(tag(pre_tags, sub.seqnum)?);
                fragment.push_str(&encoder.encode_text(&substring(
                    &src,
                    self.strict_utf16,
                    start,
                    end,
                )?));
                fragment.push_str(tag(post_tags, sub.seqnum)?);
                src_index = end;
            }
        }
        fragment.push_str(&encoder.encode_text(&substring(
            &src,
            self.strict_utf16,
            src_index,
            len_i32(&src),
        )?));
        Ok(fragment)
    }

    /// `getFragmentSourceMSO`: the text from the start boundary to the end
    /// boundary, and where it starts.
    fn fragment_source(
        &mut self,
        buffer: &mut Vec<u16>,
        index: &mut usize,
        values: &[Vec<u16>],
        start_offset: i32,
        end_offset: i32,
    ) -> Result<(Vec<u16>, i32)> {
        while i64::try_from(buffer.len()).unwrap_or(i64::MAX) < i64::from(end_offset)
            && *index < values.len()
        {
            buffer.extend_from_slice(&values[*index]);
            *index += 1;
            buffer.push(self.multi_valued_separator);
        }
        if *index == 0 {
            return Err(Error::IllegalArgument(
                "array index out of range: -1".to_string(),
            ));
        }
        // the values are tokenized text: the separator after the last is not
        // part of it
        let buffer_length = len_i32(buffer) - 1;
        let eo = if buffer_length < end_offset {
            buffer_length
        } else {
            self.boundary_scanner.find_end_offset(buffer, end_offset)
        };
        let mso = self
            .boundary_scanner
            .find_start_offset(buffer, start_offset);
        if mso < 0 || eo < mso || eo > len_i32(buffer) {
            return Err(Error::IllegalArgument(format!(
                "begin {mso}, end {eo}, length {}",
                buffer.len()
            )));
        }
        Ok((buffer[mso as usize..eo as usize].to_vec(), mso))
    }

    /// `discreteMultiValueHighlighting`: the fragments cut at the borders of
    /// the values, so none runs from one value into the next.
    fn discrete_multi_value_highlighting(
        frag_infos: Vec<WeightedFragInfo>,
        fields: &[Vec<u16>],
    ) -> Result<Vec<WeightedFragInfo>> {
        let mut result: Vec<WeightedFragInfo> = Vec::new();
        'frags: for mut frag_info in frag_infos {
            let mut field_end = 0i32;
            for field in fields {
                if field.is_empty() {
                    field_end += 1;
                    continue;
                }
                let field_start = field_end;
                // + 1 for going to next field with same name
                field_end += len_i32(field) + 1;
                if frag_info.start_offset >= field_start
                    && frag_info.end_offset >= field_start
                    && frag_info.start_offset <= field_end
                    && frag_info.end_offset <= field_end
                {
                    result.push(frag_info);
                    continue 'frags;
                }
                let Some(first_toffs) = frag_info
                    .sub_infos
                    .first()
                    .and_then(|s| s.terms_offsets.first())
                    .copied()
                else {
                    // no sub infos, or one without offsets
                    if frag_info.sub_infos.is_empty() {
                        continue 'frags;
                    }
                    // `termsOffsets().get(0)` of an empty list
                    return Err(Error::IllegalArgument(
                        "Index 0 out of bounds for length 0".to_string(),
                    ));
                };
                if frag_info.start_offset >= field_end || first_toffs.start_offset() >= field_end {
                    continue;
                }
                let mut frag_start = field_start;
                if frag_info.start_offset > field_start && frag_info.start_offset < field_end {
                    frag_start = frag_info.start_offset;
                }
                let mut frag_end = field_end;
                if frag_info.end_offset > field_start && frag_info.end_offset < field_end {
                    frag_end = frag_info.end_offset;
                }
                let mut sub_infos: Vec<SubInfo> = Vec::new();
                let mut boost = 0.0f32;
                for sub_info in &mut frag_info.sub_infos {
                    let mut toffs_list: Vec<Toffs> = Vec::new();
                    let mut kept: Vec<Toffs> = Vec::new();
                    let mut stopped_at = sub_info.terms_offsets.len();
                    for (i, toffs) in sub_info.terms_offsets.iter().enumerate() {
                        if toffs.start_offset() >= field_end {
                            stopped_at = i;
                            break;
                        }
                        let starts_after_field = toffs.start_offset() >= field_start;
                        let ends_before_field = toffs.end_offset() < field_end;
                        if starts_after_field && ends_before_field {
                            toffs_list.push(*toffs);
                        } else if starts_after_field {
                            toffs_list.push(Toffs::new(toffs.start_offset(), field_end - 1));
                            kept.push(*toffs);
                        } else if ends_before_field {
                            toffs_list.push(Toffs::new(field_start, toffs.end_offset()));
                        } else {
                            toffs_list.push(Toffs::new(field_start, field_end - 1));
                            kept.push(*toffs);
                        }
                    }
                    kept.extend_from_slice(&sub_info.terms_offsets[stopped_at..]);
                    sub_info.terms_offsets = kept;
                    if !toffs_list.is_empty() {
                        sub_infos.push(SubInfo {
                            text: sub_info.text.clone(),
                            terms_offsets: toffs_list,
                            seqnum: sub_info.seqnum,
                            boost: sub_info.boost,
                        });
                        boost += sub_info.boost;
                    }
                }
                frag_info.sub_infos.retain(|s| !s.terms_offsets.is_empty());
                result.push(WeightedFragInfo {
                    sub_infos,
                    total_boost: boost,
                    start_offset: frag_start,
                    end_offset: frag_end,
                });
            }
        }
        result.sort_by_key(|i| i.start_offset);
        Ok(result)
    }
}

/// `getPreTag`/`getPostTag`: the tag `num % tags.length` picks.
fn tag(tags: &[String], num: i32) -> Result<&str> {
    if tags.is_empty() {
        return Err(Error::IllegalArgument("/ by zero".to_string()));
    }
    let n = num.rem_euclid(i32::try_from(tags.len()).unwrap_or(i32::MAX));
    Ok(&tags[n as usize])
}

/// `String.substring(begin, end)` over UTF-16 text; an unpaired surrogate at
/// a cut reads as U+FFFD.
fn substring(text: &[u16], strict: bool, begin: i32, end: i32) -> Result<String> {
    if begin < 0 || end < begin || end > len_i32(text) {
        return Err(Error::IllegalArgument(format!(
            "begin {begin}, end {end}, length {}",
            text.len()
        )));
    }
    let units = &text[begin as usize..end as usize];
    if strict {
        return String::from_utf16(units)
            .map_err(|_| Error::Runtime("unpaired surrogate in a fragment".to_string()));
    }
    Ok(String::from_utf16_lossy(units))
}

/// `FragmentBuilderHelper.fixWeightedFragInfo`: the sub infos sorted by
/// their first match (`CollectionUtil.introSort`), and the fragment made to
/// start no later than the first of them.
fn fix_weighted_frag_info(frag_info: &mut WeightedFragInfo) {
    if frag_info.sub_infos.is_empty() {
        return;
    }
    struct ByFirstMatch<'a>(&'a mut [SubInfo], i32);
    impl ByFirstMatch<'_> {
        fn key(&self, i: usize) -> i32 {
            self.0[i]
                .terms_offsets
                .first()
                .map_or(i32::MIN, Toffs::start_offset)
        }
    }
    impl lucene_util::sorter::IntroTarget for ByFirstMatch<'_> {
        fn swap(&mut self, i: usize, j: usize) {
            self.0.swap(i, j);
        }
        fn set_pivot(&mut self, i: usize) {
            self.1 = self.key(i);
        }
        fn compare_pivot(&mut self, j: usize) -> i32 {
            self.1.cmp(&self.key(j)) as i32
        }
    }
    let n = frag_info.sub_infos.len();
    if n > 1 {
        lucene_util::sorter::intro_sort(&mut ByFirstMatch(&mut frag_info.sub_infos, 0), 0, n);
    }
    if let Some(first) = frag_info.sub_infos[0].terms_offsets.first() {
        frag_info.start_offset = frag_info.start_offset.min(first.start_offset());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sub(start: i32, seqnum: i32) -> SubInfo {
        SubInfo {
            text: String::new(),
            terms_offsets: vec![Toffs::new(start, start + 3)],
            seqnum,
            boost: 1.0,
        }
    }

    #[test]
    fn the_fix_sorts_by_first_match_and_starts_no_later() {
        let mut info = WeightedFragInfo {
            sub_infos: vec![sub(20, 0), sub(4, 1), sub(12, 2)],
            total_boost: 3.0,
            start_offset: 10,
            end_offset: 40,
        };
        fix_weighted_frag_info(&mut info);
        let starts: Vec<i32> = info
            .sub_infos
            .iter()
            .map(|s| s.terms_offsets[0].start_offset())
            .collect();
        assert_eq!(starts, vec![4, 12, 20]);
        assert_eq!((info.start_offset, info.end_offset), (4, 40));
        // a fragment without phrases is left as it is
        let mut empty = WeightedFragInfo {
            sub_infos: Vec::new(),
            total_boost: 0.0,
            start_offset: 7,
            end_offset: 9,
        };
        fix_weighted_frag_info(&mut empty);
        assert_eq!(empty.start_offset, 7);
        // many phrases: the intro sort past its insertion-sort threshold
        let mut many = WeightedFragInfo {
            sub_infos: (0..40).rev().map(|i| sub(i * 2 + 30, i)).collect(),
            total_boost: 40.0,
            start_offset: 5,
            end_offset: 200,
        };
        fix_weighted_frag_info(&mut many);
        assert!(many
            .sub_infos
            .windows(2)
            .all(|w| w[0].terms_offsets[0].start_offset() <= w[1].terms_offsets[0].start_offset()));
        assert_eq!(many.start_offset, 5);
    }

    #[test]
    fn strict_text_refuses_half_a_surrogate_pair() {
        let text: Vec<u16> = "a😀b".encode_utf16().collect();
        assert_eq!(substring(&text, false, 0, 2).unwrap(), "a\u{FFFD}");
        assert!(substring(&text, true, 0, 2).is_err());
        assert_eq!(substring(&text, true, 0, 4).unwrap(), "a😀b");
        assert!(substring(&text, true, 3, 2).is_err());
    }
}
