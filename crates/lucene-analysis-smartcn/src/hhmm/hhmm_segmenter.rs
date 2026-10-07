//! `org.apache.lucene.analysis.cn.smart.hhmm.HHMMSegmenter`: builds a
//! sentence's segment graph -- every dictionary word among its Han
//! characters, each run of letters or of digits, each delimiter, each other
//! character, each supplementary character -- between a begin and an end
//! token, and returns the shortest path through its bigram graph.

use std::sync::Arc;

use lucene_analysis::java_character::{char_count, code_point_at};
use lucene_analysis::AnalysisError;

use super::bi_seg_graph::BiSegGraph;
use super::bigram_dictionary::BigramDictionary;
use super::seg_graph::SegGraph;
use super::seg_token::SegToken;
use super::word_dictionary::WordDictionary;
use crate::utility::{
    char_type, get_char_type, word_type, END_CHAR_ARRAY, MAX_FREQUENCE, NUMBER_CHAR_ARRAY,
    START_CHAR_ARRAY, STRING_CHAR_ARRAY,
};

/// `HHMMSegmenter`, over the two dictionaries.
#[derive(Debug, Clone)]
pub struct HHMMSegmenter {
    word_dict: Arc<WordDictionary>,
    bigram_dict: Arc<BigramDictionary>,
}

impl Default for HHMMSegmenter {
    /// Over Lucene's dictionaries.
    fn default() -> Self {
        Self::new(
            WordDictionary::get_instance(),
            BigramDictionary::get_instance(),
        )
    }
}

/// A sentence offset as Java's `int` (a sentence is at most 1,024 units).
fn off(i: usize) -> i32 {
    i32::try_from(i).unwrap_or(i32::MAX)
}

impl HHMMSegmenter {
    /// A segmenter over `word_dict` and `bigram_dict`.
    pub fn new(word_dict: Arc<WordDictionary>, bigram_dict: Arc<BigramDictionary>) -> Self {
        HHMMSegmenter {
            word_dict,
            bigram_dict,
        }
    }

    /// `createSegGraph(sentence)`.
    // ARITH: i and j are sentence positions, below its length + 1.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn create_seg_graph(&self, sentence: &[u16]) -> SegGraph {
        let wd = &*self.word_dict;
        let length = sentence.len();
        let char_type_array: Vec<i32> = sentence.iter().map(|&c| get_char_type(c)).collect();
        let mut word_buf: Vec<u16> = Vec::new();
        let mut seg_graph = SegGraph::new();
        let mut i = 0;
        while i < length {
            let mut has_full_width = false;
            match char_type_array[i] {
                char_type::SPACE_LIKE => i += 1,
                char_type::SURROGATE => {
                    let state = code_point_at(sentence, i, length);
                    let count = char_count(state);
                    let token = SegToken::new(
                        &sentence[i..i + count],
                        off(i),
                        off(i + count),
                        word_type::CHINESE_WORD,
                        0,
                    );
                    seg_graph.add_token(token);
                    i += count;
                }
                char_type::HANZI => {
                    let mut j = i + 1;
                    word_buf.clear();
                    word_buf.push(sentence[i]);
                    let mut char_array: Vec<u16> = vec![sentence[i]];
                    let frequency = wd.get_frequency(&char_array);
                    seg_graph.add_token(SegToken::new(
                        &char_array,
                        off(i),
                        off(j),
                        word_type::CHINESE_WORD,
                        frequency,
                    ));
                    let mut found_index = wd.get_prefix_match(&char_array, 0);
                    while j <= length && found_index != -1 {
                        if wd.is_equal(&char_array, found_index) && char_array.len() > 1 {
                            // It is the phrase we are looking for; In other
                            // words, we have found a phrase SegToken from i
                            // to j. It is not a monosyllabic word (single
                            // word).
                            let frequency = wd.get_frequency(&char_array);
                            seg_graph.add_token(SegToken::new(
                                &char_array,
                                off(i),
                                off(j),
                                word_type::CHINESE_WORD,
                                frequency,
                            ));
                        }
                        while j < length && char_type_array[j] == char_type::SPACE_LIKE {
                            j += 1;
                        }
                        if j < length && char_type_array[j] == char_type::HANZI {
                            word_buf.push(sentence[j]);
                            char_array.clone_from(&word_buf);
                            // idArray has been found (foundWordIndex!=-1) as a
                            // prefix before. Therefore, idArray after it has
                            // been lengthened can only appear after
                            // foundWordIndex. So start searching after
                            // foundWordIndex.
                            found_index = wd.get_prefix_match(&char_array, found_index);
                            j += 1;
                        } else {
                            break;
                        }
                    }
                    i += 1;
                }
                t @ (char_type::FULLWIDTH_LETTER | char_type::LETTER) => {
                    has_full_width |= t == char_type::FULLWIDTH_LETTER;
                    let mut j = i + 1;
                    while j < length
                        && matches!(
                            char_type_array[j],
                            char_type::LETTER | char_type::FULLWIDTH_LETTER
                        )
                    {
                        has_full_width |= char_type_array[j] == char_type::FULLWIDTH_LETTER;
                        j += 1;
                    }
                    // Found a Token from i to j. Type is LETTER char string.
                    let frequency = wd.get_frequency(&STRING_CHAR_ARRAY);
                    let wt = if has_full_width {
                        word_type::FULLWIDTH_STRING
                    } else {
                        word_type::STRING
                    };
                    seg_graph.add_token(SegToken::new(
                        &STRING_CHAR_ARRAY,
                        off(i),
                        off(j),
                        wt,
                        frequency,
                    ));
                    i = j;
                }
                t @ (char_type::FULLWIDTH_DIGIT | char_type::DIGIT) => {
                    has_full_width |= t == char_type::FULLWIDTH_DIGIT;
                    let mut j = i + 1;
                    while j < length
                        && matches!(
                            char_type_array[j],
                            char_type::DIGIT | char_type::FULLWIDTH_DIGIT
                        )
                    {
                        has_full_width |= char_type_array[j] == char_type::FULLWIDTH_DIGIT;
                        j += 1;
                    }
                    // Found a Token from i to j. Type is NUMBER char string.
                    let frequency = wd.get_frequency(&NUMBER_CHAR_ARRAY);
                    let wt = if has_full_width {
                        word_type::FULLWIDTH_NUMBER
                    } else {
                        word_type::NUMBER
                    };
                    seg_graph.add_token(SegToken::new(
                        &NUMBER_CHAR_ARRAY,
                        off(i),
                        off(j),
                        wt,
                        frequency,
                    ));
                    i = j;
                }
                char_type::DELIMITER => {
                    let j = i + 1;
                    // Ignore the frequency of punctuation; set it to the
                    // largest value.
                    seg_graph.add_token(SegToken::new(
                        &sentence[i..j],
                        off(i),
                        off(j),
                        word_type::DELIMITER,
                        MAX_FREQUENCE,
                    ));
                    i = j;
                }
                _ => {
                    let j = i + 1;
                    // Treat the unrecognized char symbol as unknown string.
                    // For example, any symbol not in GB2312 is treated as one
                    // of these.
                    let frequency = wd.get_frequency(&STRING_CHAR_ARRAY);
                    seg_graph.add_token(SegToken::new(
                        &STRING_CHAR_ARRAY,
                        off(i),
                        off(j),
                        word_type::STRING,
                        frequency,
                    ));
                    i = j;
                }
            }
        }
        // Add two more Tokens: "beginning xx beginning"
        let frequency = wd.get_frequency(&START_CHAR_ARRAY);
        seg_graph.add_token(SegToken::new(
            &START_CHAR_ARRAY,
            -1,
            0,
            word_type::SENTENCE_BEGIN,
            frequency,
        ));
        // "end xx end"
        let frequency = wd.get_frequency(&END_CHAR_ARRAY);
        seg_graph.add_token(SegToken::new(
            &END_CHAR_ARRAY,
            off(length),
            off(length + 1),
            word_type::SENTENCE_END,
            frequency,
        ));
        seg_graph
    }

    /// `process(sentence)`: the segments of the lightest path, begin and
    /// end tokens included.
    pub fn process(&self, sentence: &[u16]) -> Result<Vec<SegToken>, AnalysisError> {
        let seg_graph = self.create_seg_graph(sentence);
        let bi_seg_graph = BiSegGraph::new(seg_graph, &self.bigram_dict);
        bi_seg_graph.get_short_path()
    }
}
