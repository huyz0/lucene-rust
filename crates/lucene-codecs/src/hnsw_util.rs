//! Port of `org.apache.lucene.util.hnsw.HnswUtil`: graph connectivity
//! diagnostics -- is every node on every level reachable from an entry point
//! (`isRooted`/`graphIsRooted`), and how big are the connected components
//! (`componentSizes`/`components`).
//!
//! In 10.5.0 `HnswGraphBuilder.connectComponents` is commented out, so this
//! is diagnostics only (CheckIndex-style tooling and tests); nothing on the
//! write path depends on it. It walks any [`HnswGraphView`] -- a graph being
//! built or one read off `.vex`.

use std::collections::HashSet;

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::hnsw::HnswGraphView;
use crate::vectors::{Error, Result};

/// `HnswUtil.Component`: a connected component, named by the node its walk
/// started from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    pub start: i32,
    pub size: usize,
}

fn node_index(node: i32, len: usize) -> Result<usize> {
    usize::try_from(node)
        .ok()
        .filter(|&n| n < len)
        .ok_or_else(|| Error::CorruptMeta(format!("graph node {node} outside 0..{len}")))
}

/// `HnswUtil.isRooted`: every level is a single component reachable from the
/// entry points of the level above.
pub fn is_rooted<G: HnswGraphView>(graph: &G) -> Result<bool> {
    for level in 0..graph.num_levels() {
        if components(graph, level, None, 0)?.len() > 1 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `HnswUtil.componentSizes(hnsw)`: component sizes on level 0.
pub fn component_sizes<G: HnswGraphView>(graph: &G) -> Result<Vec<usize>> {
    component_sizes_on_level(graph, 0)
}

/// `HnswUtil.componentSizes(hnsw, level)`.
pub fn component_sizes_on_level<G: HnswGraphView>(graph: &G, level: i32) -> Result<Vec<usize>> {
    Ok(components(graph, level, None, 0)?
        .into_iter()
        .map(|c| c.size)
        .collect())
}

/// `HnswUtil.components(hnsw, level, notFullyConnected, maxConn)`: the first
/// component is everything reachable from the level's entry points (the
/// graph's entry node on the top level, the next level's nodes otherwise);
/// each later one starts at the first node none of the earlier walks reached.
/// Nodes with fewer than `max_conn` neighbours are recorded in
/// `not_fully_connected` when it is given.
pub fn components<G: HnswGraphView>(
    graph: &G,
    level: i32,
    mut not_fully_connected: Option<&mut FixedBitSet>,
    max_conn: usize,
) -> Result<Vec<Component>> {
    let size = usize::try_from(graph.size()).unwrap_or(0);
    let mut connected = FixedBitSet::new(size);
    let num_levels = graph.num_levels();
    if level >= num_levels || level < 0 {
        return Err(Error::InvalidGraphParameter(format!(
            "Level {level} too large for graph with {num_levels} levels"
        )));
    }
    let entry_points: Vec<i32> = if level == num_levels.saturating_sub(1) {
        vec![graph.entry_node()]
    } else {
        graph.sorted_nodes_on_level(level.saturating_add(1))?
    };
    let mut result = Vec::new();
    let mut total = 0usize;
    for ep in entry_points {
        let c = mark_rooted(
            graph,
            level,
            &mut connected,
            not_fully_connected.as_deref_mut(),
            max_conn,
            ep,
        )?;
        total = total.saturating_add(c.size);
    }
    let first = match &not_fully_connected {
        Some(nfc) => nfc.next_set_bit(0),
        None => connected.next_set_bit(0),
    };
    if total > 0 {
        result.push(Component {
            start: first.map_or(-1, |f| f as i32),
            size: total,
        });
    }
    if level == 0 {
        let mut next_clear =
            lucene_util::fixed_bit_set::next_clear_bit_in_words(connected.words(), 0);
        while next_clear < size {
            let c = mark_rooted(
                graph,
                level,
                &mut connected,
                not_fully_connected.as_deref_mut(),
                max_conn,
                next_clear as i32,
            )?;
            result.push(c);
            next_clear =
                lucene_util::fixed_bit_set::next_clear_bit_in_words(connected.words(), next_clear);
        }
    } else {
        for node in graph.sorted_nodes_on_level(level)? {
            if connected.get(node_index(node, connected.len())?) {
                continue;
            }
            let c = mark_rooted(
                graph,
                level,
                &mut connected,
                not_fully_connected.as_deref_mut(),
                max_conn,
                node,
            )?;
            result.push(c);
        }
    }
    Ok(result)
}

/// `HnswUtil.markRooted`: a depth-first walk from `entry_point`, marking
/// what it reaches.
fn mark_rooted<G: HnswGraphView>(
    graph: &G,
    level: i32,
    connected: &mut FixedBitSet,
    mut not_fully_connected: Option<&mut FixedBitSet>,
    max_conn: usize,
    entry_point: i32,
) -> Result<Component> {
    let size = connected.len();
    if connected.get(node_index(entry_point, size)?) {
        return Ok(Component {
            start: entry_point,
            size: 0,
        });
    }
    let mut in_stack: HashSet<i32> = HashSet::new();
    let mut stack = vec![entry_point];
    let mut count = 0usize;
    let mut neighbors = Vec::new();
    while let Some(node) = stack.pop() {
        let idx = node_index(node, size)?;
        if connected.get(idx) {
            continue;
        }
        count = count.saturating_add(1);
        connected.set(idx);
        graph.neighbors_into(level, node, &mut neighbors)?;
        for &friend in &neighbors {
            if !connected.get(node_index(friend, size)?) && in_stack.insert(friend) {
                stack.push(friend);
            }
        }
        if neighbors.len() < max_conn {
            if let Some(nfc) = not_fully_connected.as_deref_mut() {
                if idx < nfc.len() {
                    nfc.set(idx);
                }
            }
        }
    }
    Ok(Component {
        start: entry_point,
        size: count,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    /// A hand-built two-level graph.
    struct TestGraph {
        levels: Vec<Vec<(i32, Vec<i32>)>>,
        size: i32,
        entry: i32,
    }

    impl HnswGraphView for TestGraph {
        fn size(&self) -> i32 {
            self.size
        }
        fn num_levels(&self) -> i32 {
            self.levels.len() as i32
        }
        fn entry_node(&self) -> i32 {
            self.entry
        }
        fn max_conn(&self) -> i32 {
            2
        }
        fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()> {
            out.clear();
            if let Some((_, n)) = self.levels[level as usize]
                .iter()
                .find(|(id, _)| *id == node)
            {
                out.extend_from_slice(n);
            }
            Ok(())
        }
        fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>> {
            Ok(self.levels[level as usize]
                .iter()
                .map(|(id, _)| *id)
                .collect())
        }
    }

    fn graph(level0: Vec<(i32, Vec<i32>)>, level1: Vec<(i32, Vec<i32>)>, entry: i32) -> TestGraph {
        let size = level0.len() as i32;
        TestGraph {
            levels: vec![level0, level1],
            size,
            entry,
        }
    }

    #[test]
    fn connected_graph_is_rooted() {
        let g = graph(
            vec![(0, vec![1]), (1, vec![0, 2]), (2, vec![1, 3]), (3, vec![2])],
            vec![(0, vec![2]), (2, vec![0])],
            0,
        );
        assert!(is_rooted(&g).unwrap());
        assert_eq!(component_sizes(&g).unwrap(), vec![4]);
        assert_eq!(component_sizes_on_level(&g, 1).unwrap(), vec![2]);
        let mut nfc = FixedBitSet::new(4);
        let c = components(&g, 0, Some(&mut nfc), 2).unwrap();
        assert_eq!(c, vec![Component { start: 0, size: 4 }]);
        // nodes 0 and 3 have one neighbour, fewer than maxConn 2.
        assert!(nfc.get(0) && nfc.get(3) && !nfc.get(1));
    }

    #[test]
    fn disconnected_graph_reports_components() {
        let g = graph(
            vec![
                (0, vec![1]),
                (1, vec![0]),
                (2, vec![3]),
                (3, vec![2]),
                (4, vec![]),
            ],
            vec![(1, vec![3]), (3, vec![])],
            1,
        );
        // Level 1: 1 -> 3, but 3 has no way back; entry 1 reaches both.
        assert_eq!(component_sizes_on_level(&g, 1).unwrap(), vec![2]);
        // Level 0: entries 1 and 3 reach {0,1} and {2,3}; node 4 is alone.
        let c = components(&g, 0, None, 0).unwrap();
        assert_eq!(
            c,
            vec![
                Component { start: 0, size: 4 },
                Component { start: 4, size: 1 }
            ]
        );
        assert!(!is_rooted(&g).unwrap());
        assert!(components(&g, 2, None, 0).is_err());
    }

    #[test]
    fn upper_level_orphan_is_its_own_component() {
        let g = graph(
            vec![(0, vec![1]), (1, vec![0]), (2, vec![])],
            vec![(0, vec![]), (2, vec![])],
            0,
        );
        let c = components(&g, 1, None, 0).unwrap();
        assert_eq!(
            c,
            vec![
                Component { start: 0, size: 1 },
                Component { start: 2, size: 1 }
            ]
        );
        let bad = TestGraph {
            levels: vec![vec![(0, vec![7])]],
            size: 1,
            entry: 0,
        };
        assert!(components(&bad, 0, None, 0).is_err());
    }
}
