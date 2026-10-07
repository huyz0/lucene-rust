//! `org.apache.lucene.analysis.cn.smart.hhmm.{BiSegGraph, SegTokenPair,
//! PathNode}`: the graph whose edges join each segment to each segment that
//! can follow it, weighted by the smoothed bigram probability, and its
//! shortest path from the sentence-begin token to the sentence-end token.
//!
//! An edge `t1 -> t2` weighs
//! `-ln(0.1 * (1 + f1) / MAX_FREQUENCE + 0.9 * ((1 - 1/MAX_FREQUENCE) * f12
//! / (1 + f1) + 1/MAX_FREQUENCE))` with `f1` the word frequency of `t1` and
//! `f12` the bigram frequency of `t1@t2`, evaluated in Java's order. Java
//! keys the edges by their target in an `IntObjectHashMap`; the targets are
//! token numbers, so here they index a vector.

use lucene_analysis::AnalysisError;

use super::bigram_dictionary::{BigramDictionary, WORD_SEGMENT_CHAR};
use super::seg_graph::SegGraph;
use super::seg_token::SegToken;
use crate::utility::MAX_FREQUENCE;

/// `SegTokenPair`: an edge. Java also keeps its `word1@word2` text, which
/// nothing reads once the bigram is looked up; it is not kept here.
#[derive(Debug, Clone, PartialEq)]
pub struct SegTokenPair {
    /// `from`: the first token's number.
    pub from: usize,
    /// `to`: the second token's number.
    pub to: usize,
    /// `weight`.
    pub weight: f64,
}

/// `PathNode`: a node's best distance and predecessor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathNode {
    pub weight: f64,
    pub pre_node: usize,
}

/// `BiSegGraph`.
#[derive(Debug)]
pub struct BiSegGraph {
    /// `tokenPairListTable`, by `to`.
    token_pair_list_table: Vec<Vec<SegTokenPair>>,
    /// `segTokenList`.
    seg_token_list: Vec<SegToken>,
}

/// Java's `NullPointerException` where a node has no incoming edge or no
/// finite path: unreachable for the segment graphs the segmenter builds.
fn no_path() -> AnalysisError {
    AnalysisError::IllegalState("NullPointerException: no path through the segment graph".into())
}

impl BiSegGraph {
    /// `new BiSegGraph(segGraph)`: numbers the segments, joins them, then
    /// takes them (Java shares them with the graph).
    pub fn new(mut seg_graph: SegGraph, bigram_dict: &BigramDictionary) -> Self {
        seg_graph.make_index();
        let mut g = BiSegGraph {
            token_pair_list_table: Vec::new(),
            seg_token_list: Vec::new(),
        };
        g.generate_bi_seg_graph(&seg_graph, bigram_dict);
        g.seg_token_list = seg_graph.into_token_list();
        g
    }

    /// `generateBiSegGraph(segGraph)`.
    fn generate_bi_seg_graph(&mut self, seg_graph: &SegGraph, bigram_dict: &BigramDictionary) {
        let smooth = 0.1;
        let tiny_double = 1.0 / f64::from(MAX_FREQUENCE);
        let max_start = seg_graph.get_max_start();
        let mut id_buffer = Vec::new();
        let mut key = -1;
        // ARITH: key and next run from -1 up to max_start, a sentence offset.
        #[allow(clippy::arithmetic_side_effects)]
        while key < max_start {
            if let Some(token_list) = seg_graph.get_start_list(key) {
                for t1 in token_list {
                    let one_word_freq = f64::from(t1.weight);
                    let mut next = t1.end_offset;
                    let mut next_tokens = None;
                    while next <= max_start {
                        if let Some(l) = seg_graph.get_start_list(next) {
                            next_tokens = Some(l);
                            break;
                        }
                        next += 1;
                    }
                    let Some(next_tokens) = next_tokens else {
                        break;
                    };
                    for t2 in next_tokens {
                        id_buffer.clear();
                        id_buffer.extend_from_slice(&t1.char_array);
                        id_buffer.push(WORD_SEGMENT_CHAR);
                        id_buffer.extend_from_slice(&t2.char_array);
                        let word_pair_freq = f64::from(bigram_dict.get_frequency(&id_buffer));
                        let weight = -(smooth * (1.0 + one_word_freq)
                            / (f64::from(MAX_FREQUENCE) + 0.0)
                            + (1.0 - smooth)
                                * ((1.0 - tiny_double) * word_pair_freq / (1.0 + one_word_freq)
                                    + tiny_double))
                            .ln();
                        self.add_seg_token_pair(SegTokenPair {
                            from: t1.index as usize,
                            to: t2.index as usize,
                            weight,
                        });
                    }
                }
            }
            key += 1;
        }
    }

    /// `isToExist(to)`.
    pub fn is_to_exist(&self, to: usize) -> bool {
        self.get_to_list(to).is_some()
    }

    /// `getToList(to)`: the edges into token `to`.
    pub fn get_to_list(&self, to: usize) -> Option<&[SegTokenPair]> {
        let l = self.token_pair_list_table.get(to)?;
        (!l.is_empty()).then_some(l.as_slice())
    }

    /// `addSegTokenPair(tokenPair)`.
    pub fn add_seg_token_pair(&mut self, token_pair: SegTokenPair) {
        let to = token_pair.to;
        if self.token_pair_list_table.len() <= to {
            self.token_pair_list_table
                .resize_with(to.saturating_add(1), Vec::new);
        }
        self.token_pair_list_table[to].push(token_pair);
    }

    /// `getToCount()`: how many tokens have an incoming edge.
    pub fn get_to_count(&self) -> usize {
        self.token_pair_list_table
            .iter()
            .filter(|l| !l.is_empty())
            .count()
    }

    /// `getShortPath()`: the tokens of the lightest path, begin and end
    /// tokens included.
    pub fn get_short_path(mut self) -> Result<Vec<SegToken>, AnalysisError> {
        let node_count = self.get_to_count();
        let mut path = Vec::with_capacity(node_count.saturating_add(1));
        path.push(PathNode {
            weight: 0.0,
            pre_node: 0,
        });
        for current in 1..=node_count {
            let edges = self.get_to_list(current).ok_or_else(no_path)?;
            let mut min_weight = f64::MAX;
            let mut min_edge = None;
            for edge in edges {
                let pre = path.get(edge.from).ok_or_else(no_path)?;
                if pre.weight + edge.weight < min_weight {
                    min_weight = pre.weight + edge.weight;
                    min_edge = Some(edge);
                }
            }
            let min_edge = min_edge.ok_or_else(no_path)?;
            path.push(PathNode {
                weight: min_weight,
                pre_node: min_edge.from,
            });
        }
        let mut current = path.len().saturating_sub(1);
        let mut rpath = vec![current];
        while current != 0 {
            current = path[current].pre_node;
            rpath.push(current);
            if rpath.len() > path.len() {
                return Err(no_path());
            }
        }
        let mut tokens: Vec<Option<SegToken>> = std::mem::take(&mut self.seg_token_list)
            .into_iter()
            .map(Some)
            .collect();
        rpath
            .iter()
            .rev()
            .map(|&id| {
                tokens
                    .get_mut(id)
                    .and_then(Option::take)
                    .ok_or_else(no_path)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_edges_are_errors() {
        let pair = |from, to, weight| SegTokenPair { from, to, weight };
        let graph = |table: Vec<Vec<SegTokenPair>>| BiSegGraph {
            token_pair_list_table: table,
            seg_token_list: vec![],
        };
        let g = graph(vec![vec![], vec![], vec![pair(0, 2, 1.0)]]);
        assert!(g.is_to_exist(2));
        assert!(!g.is_to_exist(1));
        assert!(!g.is_to_exist(9));
        // Node 1 has no edge (Java: NullPointerException).
        assert!(g.get_short_path().is_err());
        let mut g = graph(vec![]);
        g.add_seg_token_pair(pair(0, 1, f64::NAN));
        // A NaN weight: no edge is lighter than Double.MAX_VALUE.
        assert!(g.get_short_path().is_err());
        // The path's tokens are not there; an edge from a later node.
        assert!(graph(vec![vec![], vec![pair(0, 1, 1.0)]])
            .get_short_path()
            .is_err());
        assert!(graph(vec![vec![], vec![pair(7, 1, 1.0)]])
            .get_short_path()
            .is_err());
    }
}
