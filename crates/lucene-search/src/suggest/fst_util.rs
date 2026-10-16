//! `org.apache.lucene.search.suggest.analyzing.FSTUtil`: the intersection of
//! an automaton with an FST, as the set of FST nodes reached when the
//! automaton accepts.

use lucene_codecs::fst_compiler::FstOutputs;
use lucene_codecs::fst_util::{read_ceil_arc, Arc, TypedFst};
use lucene_store::{Error, Result};
use lucene_util::automaton::{Automaton, Transition, TransitionAccessor};

/// `FSTUtil.Path<T>`: where the automaton accepted.
#[derive(Debug, Clone)]
pub struct Path<V> {
    /// The automaton state.
    pub state: i32,
    /// The FST arc the path stopped at.
    pub fst_node: Arc<V>,
    /// The sum of the outputs along the path.
    pub output: V,
    /// The labels along the path.
    pub input: Vec<i32>,
}

fn corrupt(e: impl std::fmt::Display) -> Error {
    Error::Corrupted(e.to_string())
}

/// `FSTUtil.intersectPrefixPaths(a, fst)`: every path that reaches an
/// accepting state of the deterministic automaton `a` in `fst`, depth first,
/// the last-queued path first. Paths are not followed past an accepting
/// state: everything below it is accepted too.
pub fn intersect_prefix_paths<O: FstOutputs>(
    a: &Automaton,
    fst: &TypedFst<O>,
) -> Result<Vec<Path<O::Value>>> {
    let mut end_nodes = Vec::new();
    if a.get_num_states() == 0 {
        return Ok(end_nodes);
    }
    let mut queue = vec![Path {
        state: 0,
        fst_node: fst.first_arc(),
        output: O::no_output(),
        input: Vec::new(),
    }];
    let mut reader = fst.bytes_reader();
    let mut t = Transition::new();
    while let Some(path) = queue.pop() {
        if a.is_accept(path.state) {
            end_nodes.push(path);
            continue;
        }
        let count = a.init_transition(path.state, &mut t);
        for _ in 0..count {
            a.get_next_transition(&mut t);
            let (min, max) = (t.min, t.max);
            if min == max {
                if let Some(next) = fst
                    .find_target_arc(min, &path.fst_node, &mut reader)
                    .map_err(corrupt)?
                {
                    queue.push(extend::<O>(&path, t.dest, next, min));
                }
            } else {
                let mut next =
                    read_ceil_arc(min, fst, &path.fst_node, &mut reader).map_err(corrupt)?;
                while let Some(arc) = next {
                    if arc.label() > max {
                        break;
                    }
                    let label = arc.label();
                    queue.push(extend::<O>(&path, t.dest, arc.clone(), label));
                    next = if arc.is_last() {
                        None
                    } else {
                        let mut following = arc;
                        fst.read_next_real_arc(&mut following, &mut reader)
                            .map_err(corrupt)?;
                        Some(following)
                    };
                }
            }
        }
    }
    Ok(end_nodes)
}

fn extend<O: FstOutputs>(
    path: &Path<O::Value>,
    dest: i32,
    arc: Arc<O::Value>,
    label: i32,
) -> Path<O::Value> {
    let mut input = path.input.clone();
    input.push(label);
    Path {
        state: dest,
        output: O::add(&path.output, arc.output()),
        fst_node: arc,
        input,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::completion::{CompletionFieldWriter, NrtOutputs, NrtSuggester, TermDoc};
    use lucene_store::DataOutput;
    use lucene_util::automaton::automata::{make_binary, make_char_range, make_empty};
    use lucene_util::automaton::operations::concatenate;

    fn suggester(terms: &[&str]) -> NrtSuggester {
        let mut w = CompletionFieldWriter::new();
        for (i, t) in terms.iter().enumerate() {
            let mut payload = Vec::new();
            payload.write_vint(t.len() as i32);
            payload.extend_from_slice(t.as_bytes());
            payload.write_vint(6);
            payload.push(0);
            let doc = TermDoc {
                doc_id: i as i32,
                payloads: vec![payload],
            };
            w.write_term(t.as_bytes(), &[doc]).unwrap();
        }
        let mut out = Vec::new();
        assert!(w.finish(&mut out).unwrap());
        NrtSuggester::load(&out).unwrap()
    }

    fn inputs(paths: &[Path<<NrtOutputs as FstOutputs>::Value>]) -> Vec<Vec<u8>> {
        let mut v: Vec<Vec<u8>> = paths
            .iter()
            .map(|p| p.input.iter().map(|&l| l as u8).collect())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn an_exact_label_follows_one_arc() {
        let s = suggester(&["car", "cat", "dog"]);
        let paths = intersect_prefix_paths(&make_binary(b"ca"), &s.fst).unwrap();
        assert_eq!(inputs(&paths), vec![b"ca".to_vec()]);
        assert!(paths[0].state >= 0);
    }

    #[test]
    fn a_label_range_follows_every_arc_in_it() {
        let s = suggester(&["ca", "cb", "cc", "cd", "da"]);
        let a = make_binary(b"c");
        let range = make_char_range(i32::from(b'a'), i32::from(b'c'));
        let both = concatenate(&[&a, &range]);
        let paths = intersect_prefix_paths(&both, &s.fst).unwrap();
        assert_eq!(
            inputs(&paths),
            vec![b"ca".to_vec(), b"cb".to_vec(), b"cc".to_vec()]
        );
    }

    #[test]
    fn nothing_matches_an_absent_prefix_or_an_empty_automaton() {
        let s = suggester(&["cat"]);
        assert!(intersect_prefix_paths(&make_binary(b"x"), &s.fst)
            .unwrap()
            .is_empty());
        assert!(intersect_prefix_paths(&make_empty(), &s.fst)
            .unwrap()
            .is_empty());
    }
}
