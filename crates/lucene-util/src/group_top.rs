//! The best documents of the best groups: what a grouped nearest-neighbour
//! search keeps while it walks (Milvus' `group_by_field`, Qdrant's `groups`).
//!
//! [`GroupTop`] holds at most `groups` groups and at most `per_group`
//! documents in each. A group is ranked by its best document; a document of
//! a held group enters when the group has room or it beats the group's
//! worst, and a document of a new group enters when fewer than `groups` are
//! held or it beats the worst group's best, which is then dropped whole.
//! With `per_group == 1` this is the best document of each of the best
//! `groups` groups, as a block-join search keeps the best child per parent.

use std::collections::{BTreeSet, HashMap};

/// A document a [`GroupTop`] holds: ordered best last (a higher score, or on
/// a tie the lower document, compares greater), and named by its document
/// so the same document offered twice is held once.
pub trait GroupItem: Ord + Copy {
    /// The document this is.
    fn doc(&self) -> i32;
}

/// The best `per_group` documents of each of the best `groups` groups:
/// see the [module doc](self).
#[derive(Debug, Clone)]
pub struct GroupTop<T: GroupItem> {
    groups: usize,
    per_group: usize,
    /// Each held group's documents, best first.
    held: HashMap<i32, Vec<T>>,
    /// Each held group by its best document.
    ranked: BTreeSet<(T, i32)>,
    /// Each held group by its worst held document.
    floors: BTreeSet<(T, i32)>,
}

impl<T: GroupItem> GroupTop<T> {
    /// An empty top of `groups` groups of `per_group` documents (each at
    /// least one).
    pub fn new(groups: usize, per_group: usize) -> Self {
        Self {
            groups: groups.max(1),
            per_group: per_group.max(1),
            held: HashMap::new(),
            ranked: BTreeSet::new(),
            floors: BTreeSet::new(),
        }
    }

    /// The groups held.
    pub fn len(&self) -> usize {
        self.held.len()
    }

    /// Whether no group is held.
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    /// The documents held, over every group.
    pub fn docs(&self) -> usize {
        self.held.values().map(Vec::len).sum()
    }

    /// Once `groups` groups are held, the least document a search still
    /// wants: the worst document any group holds. A document below it
    /// brings in no group and displaces no document; it could still fill a
    /// group not yet full, which a search walking by score gives up for
    /// the bound.
    pub fn floor(&self) -> Option<T> {
        if self.held.len() < self.groups {
            return None;
        }
        self.floors.first().map(|&(t, _)| t)
    }

    /// Offers `item` of group `group`; whether it is now held.
    pub fn offer(&mut self, group: i32, item: T) -> bool {
        if let Some(list) = self.held.get_mut(&group) {
            let (best, worst) = ends(list);
            if let Some(at) = list.iter().position(|x| x.doc() == item.doc()) {
                if item <= list[at] {
                    return false;
                }
                list.remove(at);
            } else if list.len() >= self.per_group {
                if list.last().is_some_and(|w| item <= *w) {
                    return false;
                }
                list.pop();
            }
            let at = list.partition_point(|x| *x > item);
            list.insert(at, item);
            let (new_best, new_worst) = ends(list);
            if new_best != best {
                self.ranked.remove(&(best, group));
                self.ranked.insert((new_best, group));
            }
            if new_worst != worst {
                self.floors.remove(&(worst, group));
                self.floors.insert((new_worst, group));
            }
            return true;
        }
        if self.held.len() >= self.groups {
            let Some(&(worst_best, worst_group)) = self.ranked.first() else {
                return false;
            };
            if item <= worst_best {
                return false;
            }
            self.ranked.remove(&(worst_best, worst_group));
            if let Some(list) = self.held.remove(&worst_group) {
                let (_, w) = ends(&list);
                self.floors.remove(&(w, worst_group));
            }
        }
        self.held.insert(group, vec![item]);
        self.ranked.insert((item, group));
        self.floors.insert((item, group));
        true
    }

    /// Every held document with its group: the groups best first, each
    /// group's documents best first.
    pub fn into_groups(mut self) -> Vec<(i32, T)> {
        let mut out = Vec::with_capacity(self.docs());
        for &(_, g) in self.ranked.iter().rev() {
            if let Some(list) = self.held.remove(&g) {
                out.extend(list.into_iter().map(|t| (g, t)));
            }
        }
        out
    }

    /// Every held document, best first.
    pub fn into_sorted(self) -> Vec<T> {
        let mut out: Vec<T> = self.into_groups().into_iter().map(|(_, t)| t).collect();
        out.sort_unstable_by(|a, b| b.cmp(a));
        out
    }
}

/// A held group's best and worst documents (a held group is never empty).
fn ends<T: Copy>(list: &[T]) -> (T, T) {
    (list[0], list[list.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Hit(i32, i32);

    impl Ord for Hit {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.cmp(&o.0).then(o.1.cmp(&self.1))
        }
    }
    impl PartialOrd for Hit {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl GroupItem for Hit {
        fn doc(&self) -> i32 {
            self.1
        }
    }

    /// The exact answer: every item grouped, each group's best
    /// `per_group`, the best `groups` groups by their best.
    fn exact(items: &[(i32, Hit)], groups: usize, per_group: usize) -> Vec<Hit> {
        let mut by: HashMap<i32, Vec<Hit>> = HashMap::new();
        for &(g, h) in items {
            let list = by.entry(g).or_default();
            if let Some(at) = list.iter().position(|x| x.1 == h.1) {
                list[at] = list[at].max(h);
            } else {
                list.push(h);
            }
        }
        let mut lists: Vec<Vec<Hit>> = by
            .into_values()
            .map(|mut l| {
                l.sort_unstable_by(|a, b| b.cmp(a));
                l.truncate(per_group);
                l
            })
            .collect();
        lists.sort_unstable_by(|a, b| b[0].cmp(&a[0]));
        lists.truncate(groups);
        let mut out: Vec<Hit> = lists.into_iter().flatten().collect();
        out.sort_unstable_by(|a, b| b.cmp(a));
        out
    }

    #[test]
    fn keeps_the_best_documents_of_the_best_groups() {
        let mut t = GroupTop::new(2, 2);
        assert!(t.is_empty() && t.floor().is_none());
        assert!(t.offer(1, Hit(5, 10)));
        assert!(t.offer(1, Hit(7, 11)));
        // Group 1 is full: a worse document of it is refused.
        assert!(!t.offer(1, Hit(4, 12)));
        assert!(t.floor().is_none());
        assert!(t.offer(2, Hit(3, 20)));
        assert_eq!((t.len(), t.docs()), (2, 3));
        assert_eq!(t.floor(), Some(Hit(3, 20)));
        // A new group must beat the worst group's best.
        assert!(!t.offer(3, Hit(3, 30)));
        assert!(t.offer(3, Hit(6, 30)));
        assert_eq!(t.len(), 2);
        // The same document again: held once, at its best score.
        assert!(!t.offer(3, Hit(2, 30)));
        assert!(t.offer(3, Hit(9, 30)));
        assert_eq!(t.docs(), 3);
        // Group 1's worst displaced.
        assert!(t.offer(1, Hit(6, 13)));
        assert_eq!(
            t.clone().into_groups(),
            vec![(3, Hit(9, 30)), (1, Hit(7, 11)), (1, Hit(6, 13))]
        );
        assert_eq!(t.into_sorted(), vec![Hit(9, 30), Hit(7, 11), Hit(6, 13)]);
    }

    /// With one document per group any order finds the exact answer; with
    /// more, offered best first (as a search walking by score nearly does):
    /// a group dropped earlier then has nothing left to bring back.
    #[test]
    fn matches_the_exact_answer() {
        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as i32
        };
        for round in 0..200 {
            let n = 1 + (round % 60);
            let mut items: Vec<(i32, Hit)> = (0..n)
                .map(|_| {
                    let doc = next() % 40;
                    (doc % 7, Hit(next() % 50, doc))
                })
                .collect();
            for (groups, per_group) in [(1, 1), (3, 1), (3, 2), (5, 4), (10, 10)] {
                if per_group > 1 {
                    items.sort_by(|a, b| b.1.cmp(&a.1));
                }
                let mut t = GroupTop::new(groups, per_group);
                for &(g, h) in &items {
                    t.offer(g, h);
                }
                assert!(t.len() <= groups);
                assert_eq!(
                    t.into_sorted(),
                    exact(&items, groups, per_group),
                    "{items:?}"
                );
            }
        }
    }
}
