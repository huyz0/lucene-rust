//! `org.apache.lucene.analysis.morph.GraphvizFormatter`: the Viterbi
//! lattice of each backtraced fragment as Graphviz `dot` text, the best
//! path highlighted.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use super::connection_costs::ConnectionCosts;
use super::viterbi::{ViterbiLang, WrappedPositionArray};
use crate::AnalysisError;

const BOS_LABEL: &str = "BOS";
const EOS_LABEL: &str = "EOS";
const FONT_NAME: &str = "Helvetica";

/// `GraphvizFormatter`.
#[derive(Debug, Clone)]
pub struct GraphvizFormatter {
    costs: Arc<ConnectionCosts>,
    best_path_map: HashMap<String, String>,
    sb: String,
}

fn node_id(pos: i32, idx: impl std::fmt::Display) -> String {
    format!("{pos}.{idx}")
}

fn idx(i: i32) -> usize {
    usize::try_from(i).unwrap_or(usize::MAX)
}

impl GraphvizFormatter {
    /// `new GraphvizFormatter(costs)`.
    pub fn new(costs: Arc<ConnectionCosts>) -> Self {
        let mut sb = String::new();
        sb.push_str("digraph viterbi {\n");
        sb.push_str("  graph [ fontsize=30 labelloc=\"t\" label=\"\" splines=true overlap=false rankdir = \"LR\"];\n");
        let _ = writeln!(
            sb,
            "  edge [ fontname=\"{FONT_NAME}\" fontcolor=\"red\" color=\"#606060\" ]"
        );
        let _ = writeln!(
            sb,
            "  node [ style=\"filled\" fillcolor=\"#e8e8f0\" shape=\"Mrecord\" fontname=\"{FONT_NAME}\" ]"
        );
        sb.push_str("  init [style=invis]\n");
        let _ = writeln!(sb, "  init -> 0.0 [label=\"{BOS_LABEL}\"]");
        GraphvizFormatter {
            costs,
            best_path_map: HashMap::new(),
            sb,
        }
    }

    /// `finish()`.
    pub fn finish(&self) -> String {
        format!("{}}}", self.sb)
    }

    /// `onBacktrace(dictProvider, positions, lastBackTracePos, endPosData,
    /// fromIDX, fragment, isEnd)`.
    #[allow(clippy::too_many_arguments)]
    pub fn on_backtrace<T, L: ViterbiLang<T> + ?Sized>(
        &mut self,
        dict: &L,
        positions: &WrappedPositionArray,
        last_back_trace_pos: i32,
        end_pos: i32,
        from_idx: i32,
        fragment: &[u16],
        is_end: bool,
    ) -> Result<(), AnalysisError> {
        self.set_best_path_map(positions, last_back_trace_pos, end_pos, from_idx)?;
        let nodes = self.format_nodes(dict, positions, last_back_trace_pos, end_pos, fragment)?;
        self.sb.push_str(&nodes);
        if is_end {
            self.sb.push_str("  fini [style=invis]\n");
            self.sb.push_str("  ");
            self.sb.push_str(&node_id(end_pos, from_idx));
            let _ = write!(self.sb, " -> fini [label=\"{EOS_LABEL}\"]");
        }
        Ok(())
    }

    /// `setBestPathMap`.
    fn set_best_path_map(
        &mut self,
        positions: &WrappedPositionArray,
        start_pos: i32,
        end_pos: i32,
        from_idx: i32,
    ) -> Result<(), AnalysisError> {
        self.best_path_map.clear();
        let (mut pos, mut best_idx) = (end_pos, from_idx);
        while pos > start_pos {
            let p = positions.at(pos);
            let back_pos = p.back_pos(idx(best_idx))?;
            let back_idx = p.back_index(idx(best_idx))?;
            self.best_path_map
                .insert(node_id(back_pos, back_idx), node_id(pos, best_idx));
            pos = back_pos;
            best_idx = back_idx;
        }
        Ok(())
    }

    /// `formatNodes`.
    fn format_nodes<T, L: ViterbiLang<T> + ?Sized>(
        &self,
        dict: &L,
        positions: &WrappedPositionArray,
        start_pos: i32,
        end_pos: i32,
        fragment: &[u16],
    ) -> Result<String, AnalysisError> {
        let mut sb = String::new();
        // Output nodes
        let mut pos = start_pos.wrapping_add(1);
        while pos <= end_pos {
            let p = positions.at(pos);
            for i in 0..p.count() {
                let _ = writeln!(
                    sb,
                    "  {} [label=\"{pos}: {}\"]",
                    node_id(pos, i),
                    p.last_right_id(i)?
                );
            }
            pos = pos.wrapping_add(1);
        }
        // Output arcs
        let mut pos = end_pos;
        while pos > start_pos {
            let p = positions.at(pos);
            for i in 0..p.count() {
                let back_pos = p.back_pos(i)?;
                let back_index = p.back_index(i)?;
                let back_data = positions.at(back_pos);
                let to = node_id(pos, i);
                let from = node_id(back_pos, back_index);
                let attrs = if self.best_path_map.get(&from) == Some(&to) {
                    // This arc is on best path
                    " color=\"#40e050\" fontcolor=\"#40a050\" penwidth=3 fontsize=20"
                } else {
                    ""
                };
                let d = dict.morph_data(p.back_type(i)?);
                let back_id = p.back_id(i)?;
                let word_cost = d.word_cost(back_id);
                let bg_cost = self.costs.get(
                    back_data.last_right_id(idx(back_index))?,
                    d.left_id(back_id),
                );
                let start = idx(back_pos.wrapping_sub(start_pos));
                let len = idx(pos.wrapping_sub(back_pos));
                let surface = start
                    .checked_add(len)
                    .and_then(|end| fragment.get(start..end))
                    .map(String::from_utf16_lossy)
                    .unwrap_or_default();
                let sign = if bg_cost >= 0 { "+" } else { "" };
                let _ = writeln!(
                    sb,
                    "  {from} -> {to} [label=\"{surface} {word_cost}{sign}{bg_cost}\"{attrs}]"
                );
            }
            pos = pos.wrapping_sub(1);
        }
        Ok(sb)
    }
}
