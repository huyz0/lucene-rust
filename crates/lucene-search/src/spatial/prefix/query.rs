//! The prefix-tree queries: `AbstractPrefixTreeQuery`'s per-segment
//! `TermsEnum` traversal, `AbstractVisitingPrefixTreeQuery`'s
//! `VisitorTemplate`, and the three predicates built on them --
//! [`IntersectsPrefixTreeQuery`], [`WithinPrefixTreeQuery`] and
//! [`ContainsPrefixTreeQuery`] -- plus the scored `TermQuery` RPT makes of a
//! point on a points-only field ([`PrefixTreeTermQuery`]).
//!
//! Over a quad tree (the common case: RPT's default, heatmaps), the visitors
//! with the default `findSubCellsToVisit`/`visitScanned` -- intersects and
//! the facet counter -- run [`visit_quad`]: the same traversal with the query
//! cells as plain values (`QuadNode`) related through one
//! `QuadCellRelater`, the indexed cells read from the term bytes, and a
//! `Cell` made only for a visitor's call. Java allocates a `QuadCell`, its
//! rectangle and an iterator per node; HotSpot makes that cheap, and here it
//! was most of the time (`docs/parity.md`).
//!
//! The traversal is Java's, step for step: the query shape's cells are
//! walked depth first in term order, the field's `TermsEnum` leap-frogging
//! them with `seekCeil`, until a level is reached where the remaining terms
//! beneath a cell are scanned (`prefixGridScanLevel`). Java's `VNode` tree,
//! whose child node is one object `reset` to each sibling, is a stack of
//! owned nodes here.

use std::sync::Arc;

use lucene_codecs::blocktree::{self, FieldTerms, SeekStatus, TermsEnum};
use lucene_codecs::postings::{DocInput, LazyDocsCursor, PostingsFlags, NO_MORE_DOCS};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::spatial4j::{DistanceUtils, Shape, SpatialContext, SpatialRelation};
use lucene_util::spatial_extras::prefix_tree::{
    Cell, CellIterator, QuadCellRelater, SpatialPrefixTree,
};

use crate::collector::ScoringCollector;
use crate::document::geo::{check_walk, collect_bits, idx, set_doc};
use crate::document::{collect_live, reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

/// `AbstractPrefixTreeQuery`'s fields.
#[derive(Clone)]
pub struct PrefixTreeQueryBase {
    pub query_shape: Arc<dyn Shape>,
    pub field_name: String,
    pub grid: Arc<dyn SpatialPrefixTree>,
    pub detail_level: i32,
}

impl std::fmt::Debug for PrefixTreeQueryBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fieldName={},queryShape={},detailLevel={}",
            self.field_name, self.query_shape, self.detail_level
        )
    }
}

/// `BaseTermsEnumTraverser` (and the visitors' `thisTerm`/`indexedCell`):
/// one segment's terms of the field, the current term, and its cell.
pub(crate) struct Traverser<'a> {
    terms: Option<TermsEnum<'a>>,
    field: Option<&'a FieldTerms>,
    doc_in: Option<&'a DocInput<'a>>,
    /// `postingsEnum`, reused from term to term (`termsEnum.postings(
    /// postingsEnum, NONE)`).
    reuse: Option<LazyDocsCursor<'a>>,
    grid: &'a dyn SpatialPrefixTree,
    pub(crate) max_doc: i32,
    /// `thisTerm != null`: `false` once the terms are exhausted.
    pub(crate) on_term: bool,
    /// `indexedCell`: the cell of the current term (the last one read).
    pub(crate) indexed_cell: Option<Box<dyn Cell>>,
    /// `indexed_cell` is an earlier term's: the term moved without reading
    /// its cell ([`Self::next_term_raw`]); [`Self::fresh`] reads it.
    stale: bool,
    /// The first postings doc id outside the segment (a corrupt `.doc`).
    pub(crate) bad: Option<i32>,
}

/// A term's documents (see [`Traverser::term_docs`]).
enum TermDocs<'c, 'a> {
    /// A pulsed single document.
    One(i32),
    /// A posting list, through the reused cursor.
    Many(&'c mut LazyDocsCursor<'a>),
}

impl<'a> Traverser<'a> {
    pub(crate) fn new(
        leaf: &OpenSegment<'a>,
        field: &str,
        grid: &'a dyn SpatialPrefixTree,
    ) -> Result<Self> {
        let max_doc = reader(leaf)?.max_doc;
        let field = leaf.fields.field(field);
        Ok(Traverser {
            terms: field.map(FieldTerms::iter),
            field,
            doc_in: leaf.doc_in,
            reuse: None,
            grid,
            max_doc,
            on_term: false,
            indexed_cell: None,
            stale: false,
            bad: None,
        })
    }

    /// Whether the field has terms in this segment (`termsEnum != null`).
    pub(crate) fn has_terms(&self) -> bool {
        self.terms.is_some()
    }

    /// The current indexed cell; only after a successful advance.
    pub(crate) fn cell(&self) -> &dyn Cell {
        self.indexed_cell
            .as_deref()
            .expect("an indexed cell is read with each term")
    }

    /// `indexedCell = grid.readCell(thisTerm, indexedCell)`: the scratch
    /// cell reused.
    fn read_current(&mut self) -> Result<()> {
        let term = self
            .terms
            .as_ref()
            .and_then(TermsEnum::term)
            .expect("positioned on a term");
        match &mut self.indexed_cell {
            Some(cell) => self.grid.read_cell_into(term, cell)?,
            None => self.indexed_cell = Some(self.grid.read_cell(term)?),
        }
        self.on_term = true;
        self.stale = false;
        Ok(())
    }

    /// The current term's bytes; only on a term.
    fn term(&self) -> &[u8] {
        self.terms
            .as_ref()
            .and_then(TermsEnum::term)
            .expect("positioned on a term")
    }

    /// [`Self::next_term`] without reading the term's cell, for a caller
    /// that reads the term itself and calls [`Self::fresh`] before anything
    /// asks for [`Self::cell`].
    fn next_term_raw(&mut self) -> Result<bool> {
        let Some(terms) = self.terms.as_mut() else {
            self.on_term = false;
            return Ok(false);
        };
        if terms.try_next_term()?.is_none() {
            self.on_term = false;
            return Ok(false);
        }
        self.on_term = true;
        self.stale = true;
        Ok(true)
    }

    /// [`Self::try_seek_ceil`] without reading the cell the term lands on
    /// (see [`Self::next_term_raw`]).
    fn try_seek_ceil_raw(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let terms = self
            .terms
            .as_mut()
            .expect("seekCeil only on a segment with terms");
        let status = terms.try_seek_ceil(target)?;
        if status == SeekStatus::End {
            self.on_term = false;
            return Ok(status);
        }
        self.on_term = true;
        self.stale = true;
        Ok(status)
    }

    /// Reads the current term's cell if [`Self::next_term_raw`] skipped it.
    fn fresh(&mut self) -> Result<()> {
        if self.stale && self.on_term {
            self.read_current()?;
        }
        self.stale = false;
        Ok(())
    }

    /// `nextTerm()`: the next term and its cell; `false` at the end.
    pub(crate) fn next_term(&mut self) -> Result<bool> {
        let Some(terms) = self.terms.as_mut() else {
            self.on_term = false;
            return Ok(false);
        };
        if terms.try_next_term()?.is_none() {
            self.on_term = false;
            return Ok(false);
        }
        self.read_current()?;
        Ok(true)
    }

    /// `termsEnum.seekCeil(target)`, then the term it lands on and its cell
    /// (`on_term` is `false` at the end).
    pub(crate) fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let terms = self
            .terms
            .as_mut()
            .expect("seekCeil only on a segment with terms");
        let status = terms.try_seek_ceil(target)?;
        if status == SeekStatus::End {
            self.on_term = false;
            return Ok(status);
        }
        self.read_current()?;
        Ok(status)
    }

    /// Runs `f` with the current indexed cell taken out of the traverser
    /// (so `f` may read postings), then puts it back -- Java's visitors
    /// hold `indexedCell` while they collect.
    pub(crate) fn with_cell<R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut dyn Cell) -> Result<R>,
    ) -> Result<R> {
        let mut cell = self.indexed_cell.take().expect("an indexed cell is read");
        let r = f(self, &mut *cell);
        self.indexed_cell = Some(cell);
        r
    }

    /// `termsEnum.docFreq()`.
    pub(crate) fn doc_freq(&mut self) -> Result<i32> {
        let terms = self.terms.as_mut().expect("docFreq only on a term");
        Ok(terms.try_stats()?.map_or(0, |s| s.doc_freq))
    }

    /// The current term's documents, deleted ones included
    /// (`termsEnum.postings(postingsEnum, NONE)`): its one document pulsed
    /// into the term's metadata, or its postings through the cursor reused
    /// from term to term. `None` off a term.
    fn term_docs(&mut self) -> Result<Option<TermDocs<'_, 'a>>> {
        let terms = self.terms.as_mut().expect("postings only on a term");
        let Some(term) = terms.try_seeked_term()? else {
            return Ok(None);
        };
        if let Some(doc) = term.singleton_doc() {
            return Ok(Some(TermDocs::One(doc)));
        }
        let field = self.field.expect("a segment with terms has the field");
        let doc_in = self.doc_in.ok_or(blocktree::Error::Unsupported(
            "postings() needs an opened .doc file for docFreq > 1 terms",
        ))?;
        let cursor =
            field.reuse_postings_for(&term, doc_in, PostingsFlags::DocsOnly, &mut self.reuse)?;
        Ok(Some(TermDocs::Many(cursor)))
    }

    /// The current term's documents, deleted ones included, each passed to
    /// `f` until it returns `false`: Java's `postingsEnum` loop.
    pub(crate) fn for_each_doc(&mut self, mut f: impl FnMut(i32) -> bool) -> Result<()> {
        match self.term_docs()? {
            None => {}
            Some(TermDocs::One(doc)) => {
                f(doc);
            }
            Some(TermDocs::Many(cursor)) => loop {
                let doc = cursor.next_doc().map_err(blocktree::Error::Postings)?;
                if doc == NO_MORE_DOCS || !f(doc) {
                    break;
                }
            },
        }
        Ok(())
    }

    /// `collectDocs(bitSet)` / `collectDocs(docSetBuilder)`: Java's
    /// `bitSet.or(postingsEnum)`, a posting list ORed in a block at a time
    /// (`intoBitSet`). A document outside the segment -- a corrupt `.doc`;
    /// Java's `FixedBitSet` throws -- is remembered in `bad`, as
    /// [`set_doc`] does.
    pub(crate) fn collect_docs(&mut self, bits: &mut FixedBitSet) -> Result<()> {
        let max_doc = self.max_doc;
        let bad = match self.term_docs()? {
            None => None,
            Some(TermDocs::One(doc)) => {
                let mut bad = None;
                set_doc(bits, doc, &mut bad);
                bad
            }
            Some(TermDocs::Many(cursor)) => {
                let len = bits.len();
                let end = i32::try_from(len).unwrap_or(i32::MAX).min(max_doc);
                let mut words = std::mem::replace(bits, FixedBitSet::new(0)).into_words();
                let r = (cursor.next_doc()).and_then(|_| cursor.into_window(0, end, &mut words));
                *bits = FixedBitSet::from_words(words, len);
                // the first document the window did not take
                Some(r.map_err(blocktree::Error::Postings)?).filter(|&d| d != NO_MORE_DOCS)
            }
        };
        if let Some(doc) = bad {
            self.bad.get_or_insert(doc);
        }
        Ok(())
    }

    /// The error for a postings doc id outside the segment, if one was met.
    pub(crate) fn check(&self, field: &str) -> Result<()> {
        check_walk(self.bad, field, self.max_doc)
    }
}

/// `AbstractVisitingPrefixTreeQuery`'s fields: the base and the scan level.
#[derive(Clone, Debug)]
pub struct VisitingQuery {
    pub base: PrefixTreeQueryBase,
    /// `prefixGridScanLevel`, clamped to `0..maxLevels-1`.
    pub prefix_grid_scan_level: i32,
}

impl VisitingQuery {
    /// `new AbstractVisitingPrefixTreeQuery(..)`.
    pub fn new(
        query_shape: Arc<dyn Shape>,
        field_name: &str,
        grid: Arc<dyn SpatialPrefixTree>,
        detail_level: i32,
        prefix_grid_scan_level: i32,
    ) -> Self {
        let max_levels = grid.max_levels();
        VisitingQuery {
            prefix_grid_scan_level: prefix_grid_scan_level.min(max_levels - 1).max(0),
            base: PrefixTreeQueryBase {
                query_shape,
                field_name: field_name.to_string(),
                grid,
                detail_level,
            },
        }
    }
}

/// `VisitorTemplate`'s abstract methods (and its overridable ones).
pub(crate) trait Visitor {
    /// `start()`.
    fn start(&mut self, t: &mut Traverser<'_>) -> Result<()>;

    /// `findSubCellsToVisit(cell)`: the query shape's cells beneath `cell`.
    fn find_sub_cells_to_visit(
        &mut self,
        q: &VisitingQuery,
        cell: &dyn Cell,
    ) -> Result<Box<dyn CellIterator>> {
        Ok(cell.next_level_cells(Some(&q.base.query_shape))?)
    }

    /// `visitPrefix(cell)`: whether to descend.
    fn visit_prefix(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<bool>;

    /// `visitLeaf(cell)`.
    fn visit_leaf(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<()>;

    /// Whether [`Self::visit_scanned`] is the default one, which the scan
    /// may answer from a [`QuadCellRelater`] without reading each cell.
    fn scans_by_default(&self) -> bool {
        true
    }

    /// [`Self::visit_scanned`]'s `visitLeaf`/`visitPrefix` for a quad cell
    /// the scan found intersecting (`relate`), for a visitor that needs only
    /// the cell's level, leaf flag and relation: `true` when done, `false`
    /// to be handed the cell itself.
    fn visit_scanned_quad(
        &mut self,
        _q: &VisitingQuery,
        _t: &mut Traverser<'_>,
        _level: i32,
        _leaf: bool,
        _relate: SpatialRelation,
    ) -> Result<bool> {
        Ok(false)
    }

    /// `visitScanned(cell)`: a leaf or a cell at the detail level met while
    /// scanning, visited when it intersects the query shape.
    fn visit_scanned(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &mut dyn Cell,
    ) -> Result<()> {
        let relate = cell.relate_shape(&*q.base.query_shape)?;
        if relate.intersects() {
            cell.set_shape_rel(Some(relate)); // just being pedantic
            if cell.is_leaf() {
                self.visit_leaf(q, t, cell)?;
            } else {
                self.visit_prefix(q, t, cell)?;
            }
        }
        Ok(())
    }
}

/// A `VNode`: a query cell and, once descended into, its children.
struct VNode {
    cell: Box<dyn Cell>,
    children: Option<Box<dyn CellIterator>>,
}

/// `VisitorTemplate.getDocIdSet()`: the traversal. `false` when the
/// segment has no terms of the field (Java's `null` before `start()`).
pub(crate) fn visit(q: &VisitingQuery, t: &mut Traverser<'_>, v: &mut dyn Visitor) -> Result<bool> {
    if v.scans_by_default() && q.base.query_shape.as_point().is_none() {
        if let Some(relater) = q.base.grid.quad_relater(&q.base.query_shape) {
            if relater.max_levels() < QUAD_MAX_BYTES as i32 {
                return visit_quad(q, t, v, relater);
            }
        }
    }
    if !t.has_terms() || !t.next_term()? {
        return Ok(false);
    }
    // The stack's last node is `curVNode`; each node's parent is the one
    // below it.
    let mut stack = vec![VNode {
        cell: q.base.grid.world_cell(),
        children: None,
    }];
    // `curVNodeTerm`: one buffer for every seek target.
    let mut target = Vec::new();
    // The scan's cell relations from the terms' bytes, for a quad tree.
    let mut scan = Scan {
        relater: if v.scans_by_default() {
            q.base.grid.quad_relater(&q.base.query_shape)
        } else {
            None
        },
        prefix: Vec::new(),
    };
    v.start(t)?;
    add_intersecting_children(q, t, v, &mut stack, &mut scan)?;

    'main: while t.on_term {
        // Advance curVNode pointer
        let top = stack.len() - 1;
        if stack[top].children.is_some() {
            // -- HAVE CHILDREN: DESCEND
            let children = stack[top].children.as_mut().expect("checked");
            let cell = children.next_detached()?;
            stack.push(VNode {
                cell,
                children: None,
            });
        } else {
            // -- NO CHILDREN: ADVANCE TO NEXT SIBLING
            stack.pop();
            loop {
                let Some(parent) = stack.last_mut() else {
                    break 'main; // all done
                };
                let children = parent
                    .children
                    .as_mut()
                    .expect("a parent node has its children");
                if children.has_next()? {
                    let cell = children.next_detached()?;
                    stack.push(VNode {
                        cell,
                        children: None,
                    });
                    break;
                }
                // reached end of siblings; pop up
                stack.pop();
            }
        }

        let cur = stack.len() - 1;
        // Seek to curVNode's cell (or skip if termsEnum has moved beyond)
        let compare = t.cell().compare_to_no_leaf(&*stack[cur].cell);
        if compare > 0 {
            // The indexed cell is after; continue loop to next query cell
            continue;
        }
        if compare < 0 {
            // The indexed cell is before; seek ahead to query cell
            stack[cur].cell.token_bytes_no_leaf_into(&mut target);
            let status = t.try_seek_ceil(&target)?;
            if status == SeekStatus::End {
                break; // all done
            }
            if status == SeekStatus::NotFound {
                // Did we find a leaf of the cell we were looking for or
                // something after?
                let cell = t.cell();
                if !cell.is_leaf() || cell.compare_to_no_leaf(&*stack[cur].cell) != 0 {
                    continue; // The indexed cell is after
                }
            }
        }
        // indexedCell == queryCell (disregarding leaf).

        // If indexedCell is a leaf then there's no prefix (prefix sorts
        // before) -- just visit and continue
        if t.cell().is_leaf() {
            t.with_cell(|t, cell| v.visit_leaf(q, t, cell))?;
            if !t.next_term()? {
                break;
            }
            continue;
        }
        // If a prefix (non-leaf) then visit; see if we descend. (The query
        // cell, not the indexed one.)
        let descend = v.visit_prefix(q, t, &*stack[cur].cell)?;
        if !t.next_term()? {
            break;
        }
        // Check for adjacent leaf with the same prefix
        if t.cell().is_leaf() && t.cell().level() == stack[cur].cell.level() {
            t.with_cell(|t, cell| v.visit_leaf(q, t, cell))?;
            if !t.next_term()? {
                break;
            }
        }

        if descend {
            add_intersecting_children(q, t, v, &mut stack, &mut scan)?;
        }
    }
    Ok(true)
}

/// `addIntersectingChildren()`: divide and conquer below `curVNode`, or
/// scan its terms once at the scan level.
fn add_intersecting_children(
    q: &VisitingQuery,
    t: &mut Traverser<'_>,
    v: &mut dyn Visitor,
    stack: &mut [VNode],
    scan_state: &mut Scan,
) -> Result<()> {
    let cur = stack.len() - 1;
    let level = stack[cur].cell.level();
    if level >= q.base.detail_level {
        return Err(Error::IllegalState("Spatial logic error".into()));
    }
    // Scanning is a performance optimization trade-off.
    let scan = level >= q.prefix_grid_scan_level; // simple heuristic
    if !scan {
        // Divide & conquer (ultimately termsEnum.seek())
        let mut sub_cells = v.find_sub_cells_to_visit(q, &*stack[cur].cell)?;
        if !sub_cells.has_next()? {
            return Ok(()); // not expected
        }
        stack[cur].children = Some(sub_cells);
    } else {
        // Scan (loop of termsEnum.next())
        match &mut scan_state.relater {
            Some(r) => scan_quad_terms(
                q,
                t,
                v,
                &*stack[cur].cell,
                q.base.detail_level,
                r,
                &mut scan_state.prefix,
            )?,
            None => scan_terms(q, t, v, &*stack[cur].cell, q.base.detail_level)?,
        }
    }
    Ok(())
}

/// `scan(scanDetailLevel)`: the terms beneath `cur`, a leaf or a cell at
/// `scan_detail_level` each visited.
fn scan_terms(
    q: &VisitingQuery,
    t: &mut Traverser<'_>,
    v: &mut dyn Visitor,
    cur: &dyn Cell,
    scan_detail_level: i32,
) -> Result<()> {
    while cur.is_prefix_of(t.cell()) {
        let level = t.cell().level();
        if level == scan_detail_level || (level < scan_detail_level && t.cell().is_leaf()) {
            t.with_cell(|t, cell| v.visit_scanned(q, t, cell))?;
        }
        // advance
        if !t.next_term()? {
            break;
        }
    }
    Ok(())
}

/// What [`scan_quad_terms`] keeps across scans: the relater and the scan
/// cell's bytes.
struct Scan {
    relater: Option<QuadCellRelater>,
    prefix: Vec<u8>,
}

/// [`scan_terms`] over a quad tree, for a visitor with the default
/// `visitScanned`: each term's level, leaf flag and relation to the query
/// shape come from its bytes ([`QuadCellRelater`]), and its cell is read
/// only when the visitor is called -- for a cell that intersects. The same
/// cells are visited, with the same relations, as [`scan_terms`] visits.
fn scan_quad_terms(
    q: &VisitingQuery,
    t: &mut Traverser<'_>,
    v: &mut dyn Visitor,
    cur: &dyn Cell,
    scan_detail_level: i32,
    relater: &mut QuadCellRelater,
    prefix: &mut Vec<u8>,
) -> Result<()> {
    cur.token_bytes_no_leaf_into(prefix);
    scan_quad_prefix(q, t, v, prefix, scan_detail_level, relater)
}

/// [`scan_quad_terms`] beneath the cell whose token is `prefix`.
fn scan_quad_prefix(
    q: &VisitingQuery,
    t: &mut Traverser<'_>,
    v: &mut dyn Visitor,
    prefix: &[u8],
    scan_detail_level: i32,
    relater: &mut QuadCellRelater,
) -> Result<()> {
    loop {
        let (bytes, leaf) = relater.split_term(t.term());
        // `bytes.starts_with(prefix)`, inline: a token is a few bytes
        if bytes.len() < prefix.len() || bytes.iter().zip(prefix).any(|(a, b)| a != b) {
            break;
        }
        let level = i32::try_from(bytes.len()).unwrap_or(i32::MAX);
        if level == scan_detail_level || (level < scan_detail_level && leaf) {
            let relate = relater.relate(bytes)?;
            if relate.intersects() && !v.visit_scanned_quad(q, t, level, leaf, relate)? {
                t.fresh()?;
                t.with_cell(|t, cell| {
                    cell.set_shape_rel(Some(relate)); // just being pedantic
                    if cell.is_leaf() {
                        v.visit_leaf(q, t, cell)
                    } else {
                        v.visit_prefix(q, t, cell).map(|_| ())
                    }
                })?;
            }
        }
        // advance
        if !t.next_term_raw()? {
            break;
        }
    }
    t.fresh()
}

/// The deepest quad cell token [`visit_quad`] keeps inline (one byte a
/// level, below `QuadPrefixTree.MAX_LEVELS_POSSIBLE`'s 50).
const QUAD_MAX_BYTES: usize = 56;

/// A query cell of [`visit_quad`]: a quad cell as plain values -- its
/// token, leaf flag and relation to the query shape -- where the generic
/// traversal keeps a boxed `LegacyCell`.
#[derive(Clone, Copy)]
struct QuadNode {
    bytes: [u8; QUAD_MAX_BYTES],
    len: usize,
    leaf: bool,
    rel: Option<SpatialRelation>,
}

impl QuadNode {
    fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

/// `getNextLevelCells(queryShape)` of a [`QuadNode`]: the next label to
/// try and the child `hasNext()` found.
struct QuadChildren {
    next_label: u8,
    pending: Option<QuadNode>,
}

/// A `VNode` of [`visit_quad`].
struct QuadVNode {
    node: QuadNode,
    children: Option<QuadChildren>,
}

/// `FilterCellIterator.hasNext()` over a quad cell's children `A`-`D`
/// (`LegacyCell.getNextLevelCells(shapeFilter)`): the next child whose
/// rectangle intersects the query shape, with its relation, made a leaf
/// when `WITHIN` or at the last level.
fn quad_has_next(
    parent: &QuadNode,
    it: &mut QuadChildren,
    relater: &mut QuadCellRelater,
) -> Result<bool> {
    if it.pending.is_some() {
        return Ok(true);
    }
    while it.next_label < 4 {
        let mut child = *parent;
        child.bytes[parent.len] = b'A' + it.next_label;
        child.len = parent.len + 1;
        it.next_label += 1;
        let rel = relater.relate(child.bytes())?;
        if rel.intersects() {
            child.rel = Some(rel);
            child.leaf = rel == SpatialRelation::Within
                || i32::try_from(child.len).unwrap_or(i32::MAX) == relater.max_levels();
            it.pending = Some(child);
            return Ok(true);
        }
    }
    Ok(false)
}

/// [`visit`] for a quad tree and a visitor with the default
/// `findSubCellsToVisit`/`visitScanned`: the same traversal, step for
/// step, with the query cells as [`QuadNode`] values related through one
/// [`QuadCellRelater`], and the indexed cells read from the term bytes --
/// a cell (a `LegacyCell`) is made only for the visitor's calls. The same
/// cells are visited in the same order with the same relations.
fn visit_quad(
    q: &VisitingQuery,
    t: &mut Traverser<'_>,
    v: &mut dyn Visitor,
    mut relater: QuadCellRelater,
) -> Result<bool> {
    if !t.has_terms() || !t.next_term_raw()? {
        return Ok(false);
    }
    let grid = &*q.base.grid;
    // The query cell handed to `visitPrefix`, refilled per call.
    let mut query_cell = grid.world_cell();
    let world = QuadNode {
        bytes: [0; QUAD_MAX_BYTES],
        len: 0,
        leaf: relater.max_levels() == 0,
        rel: None,
    };
    let mut stack = vec![QuadVNode {
        node: world,
        children: None,
    }];
    v.start(t)?;
    quad_add_intersecting_children(q, t, v, &mut stack, &mut relater)?;

    'main: while t.on_term {
        // Advance curVNode pointer
        let top = stack.len() - 1;
        if let Some(children) = &mut stack[top].children {
            // -- HAVE CHILDREN: DESCEND
            let node = children.pending.take().expect("hasNext() found the child");
            stack.push(QuadVNode {
                node,
                children: None,
            });
        } else {
            // -- NO CHILDREN: ADVANCE TO NEXT SIBLING
            stack.pop();
            loop {
                let Some(parent) = stack.last_mut() else {
                    break 'main; // all done
                };
                let node = parent.node;
                let children = parent
                    .children
                    .as_mut()
                    .expect("a parent node has its children");
                if quad_has_next(&node, children, &mut relater)? {
                    let node = children.pending.take().expect("hasNext() found it");
                    stack.push(QuadVNode {
                        node,
                        children: None,
                    });
                    break;
                }
                // reached end of siblings; pop up
                stack.pop();
            }
        }

        let cur = stack[stack.len() - 1].node;
        // Seek to curVNode's cell (or skip if termsEnum has moved beyond)
        let compare = relater.split_term(t.term()).0.cmp(cur.bytes());
        if compare.is_gt() {
            // The indexed cell is after; continue loop to next query cell
            continue;
        }
        if compare.is_lt() {
            // The indexed cell is before; seek ahead to query cell
            let status = t.try_seek_ceil_raw(cur.bytes())?;
            if status == SeekStatus::End {
                break; // all done
            }
            if status == SeekStatus::NotFound {
                // Did we find a leaf of the cell we were looking for or
                // something after?
                let (bytes, leaf) = relater.split_term(t.term());
                if !leaf || bytes != cur.bytes() {
                    continue; // The indexed cell is after
                }
            }
        }
        // indexedCell == queryCell (disregarding leaf).

        // If indexedCell is a leaf then there's no prefix (prefix sorts
        // before) -- just visit and continue
        if relater.split_term(t.term()).1 {
            t.fresh()?;
            t.with_cell(|t, cell| v.visit_leaf(q, t, cell))?;
            if !t.next_term_raw()? {
                break;
            }
            continue;
        }
        // If a prefix (non-leaf) then visit; see if we descend. (The query
        // cell, not the indexed one.)
        grid.read_cell_into(cur.bytes(), &mut query_cell)?;
        query_cell.set_shape_rel(cur.rel);
        if cur.leaf {
            query_cell.set_leaf();
        }
        let descend = v.visit_prefix(q, t, &*query_cell)?;
        if !t.next_term_raw()? {
            break;
        }
        // Check for adjacent leaf with the same prefix
        let (bytes, leaf) = relater.split_term(t.term());
        if leaf && bytes.len() == cur.len {
            t.fresh()?;
            t.with_cell(|t, cell| v.visit_leaf(q, t, cell))?;
            if !t.next_term_raw()? {
                break;
            }
        }

        if descend {
            quad_add_intersecting_children(q, t, v, &mut stack, &mut relater)?;
        }
    }
    t.fresh()?;
    Ok(true)
}

/// `addIntersectingChildren()` for [`visit_quad`].
fn quad_add_intersecting_children(
    q: &VisitingQuery,
    t: &mut Traverser<'_>,
    v: &mut dyn Visitor,
    stack: &mut [QuadVNode],
    relater: &mut QuadCellRelater,
) -> Result<()> {
    let cur = stack.len() - 1;
    let node = stack[cur].node;
    let level = i32::try_from(node.len).unwrap_or(i32::MAX);
    if level >= q.base.detail_level {
        return Err(Error::IllegalState("Spatial logic error".into()));
    }
    // Scanning is a performance optimization trade-off.
    let scan = level >= q.prefix_grid_scan_level; // simple heuristic
    if !scan {
        // Divide & conquer (ultimately termsEnum.seek())
        let mut children = QuadChildren {
            next_label: 0,
            pending: None,
        };
        if !quad_has_next(&node, &mut children, relater)? {
            return Ok(()); // not expected
        }
        stack[cur].children = Some(children);
    } else {
        // Scan (loop of termsEnum.next())
        scan_quad_prefix(q, t, v, node.bytes(), q.base.detail_level, relater)?;
    }
    Ok(())
}

/// A visiting query's segment bitset, its live documents collected at
/// `boost`.
fn collect_set(
    leaf: &OpenSegment<'_>,
    bits: Option<&FixedBitSet>,
    boost: f32,
    collector: &mut dyn ScoringCollector,
) {
    if let Some(bits) = bits {
        collect_bits(leaf, bits, boost, collector);
    }
}

/// `IntersectsPrefixTreeQuery`: documents with a shape that is not
/// disjoint from the query shape.
#[derive(Clone, Debug)]
pub struct IntersectsPrefixTreeQuery {
    pub q: VisitingQuery,
}

impl IntersectsPrefixTreeQuery {
    pub fn new(
        query_shape: Arc<dyn Shape>,
        field_name: &str,
        grid: Arc<dyn SpatialPrefixTree>,
        detail_level: i32,
        prefix_grid_scan_level: i32,
    ) -> Self {
        IntersectsPrefixTreeQuery {
            q: VisitingQuery::new(
                query_shape,
                field_name,
                grid,
                detail_level,
                prefix_grid_scan_level,
            ),
        }
    }

    /// `getDocIdSet(context)`: `None` when the segment has no terms.
    pub(crate) fn doc_id_set(&self, leaf: &OpenSegment<'_>) -> Result<Option<FixedBitSet>> {
        let mut t = Traverser::new(leaf, &self.q.base.field_name, &*self.q.base.grid)?;
        let mut v = IntersectsVisitor { results: None };
        let had = visit(&self.q, &mut t, &mut v)?;
        t.check(&self.q.base.field_name)?;
        Ok(if had { v.results } else { None })
    }
}

struct IntersectsVisitor {
    results: Option<FixedBitSet>,
}

impl IntersectsVisitor {
    fn results(&mut self) -> &mut FixedBitSet {
        self.results.as_mut().expect("start() made the results")
    }
}

impl Visitor for IntersectsVisitor {
    fn start(&mut self, t: &mut Traverser<'_>) -> Result<()> {
        self.results = Some(FixedBitSet::new(idx(t.max_doc)));
        Ok(())
    }

    fn visit_prefix(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<bool> {
        if cell.shape_rel() == Some(SpatialRelation::Within) || cell.level() == q.base.detail_level
        {
            t.collect_docs(self.results())?;
            return Ok(false);
        }
        Ok(true)
    }

    fn visit_leaf(
        &mut self,
        _q: &VisitingQuery,
        t: &mut Traverser<'_>,
        _cell: &dyn Cell,
    ) -> Result<()> {
        t.collect_docs(self.results())
    }

    /// `visitLeaf` collects; `visitPrefix` collects a `WITHIN` cell or one
    /// at the detail level -- which a scanned non-leaf always is.
    fn visit_scanned_quad(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        level: i32,
        leaf: bool,
        relate: SpatialRelation,
    ) -> Result<bool> {
        if leaf || relate == SpatialRelation::Within || level == q.base.detail_level {
            t.collect_docs(self.results())?;
        }
        Ok(true)
    }
}

impl DocumentQuery for IntersectsPrefixTreeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        collect_set(leaf, self.doc_id_set(leaf)?.as_ref(), boost, collector);
        Ok(())
    }
}

/// `WithinPrefixTreeQuery`: documents whose shape is within the query
/// shape -- found by also visiting the cells outside it (all of them, or
/// those within a buffer of it) and excluding their documents.
#[derive(Clone, Debug)]
pub struct WithinPrefixTreeQuery {
    pub q: VisitingQuery,
    /// `bufferedQueryShape`; `None` for the whole world.
    pub buffered_query_shape: Option<Arc<dyn Shape>>,
}

impl WithinPrefixTreeQuery {
    /// `new WithinPrefixTreeQuery(.., queryBuffer)`: `-1` examines the whole
    /// world.
    ///
    /// # Errors
    /// A non-positive buffer other than `-1`, or the buffering's geometry
    /// error.
    pub fn new(
        query_shape: Arc<dyn Shape>,
        field_name: &str,
        grid: Arc<dyn SpatialPrefixTree>,
        detail_level: i32,
        prefix_grid_scan_level: i32,
        query_buffer: f64,
    ) -> Result<Self> {
        #[allow(clippy::float_cmp)] // Java's exact flag value
        let buffered_query_shape = if query_buffer == -1.0 {
            None
        } else {
            Some(buffer_shape(
                grid.spatial_context(),
                &query_shape,
                query_buffer,
            )?)
        };
        Ok(WithinPrefixTreeQuery {
            q: VisitingQuery::new(
                query_shape,
                field_name,
                grid,
                detail_level,
                prefix_grid_scan_level,
            ),
            buffered_query_shape,
        })
    }

    pub(crate) fn doc_id_set(&self, leaf: &OpenSegment<'_>) -> Result<Option<FixedBitSet>> {
        let mut t = Traverser::new(leaf, &self.q.base.field_name, &*self.q.base.grid)?;
        let mut v = WithinVisitor {
            inside: None,
            outside: None,
            buffered: self.buffered_query_shape.clone(),
        };
        let had = visit(&self.q, &mut t, &mut v)?;
        t.check(&self.q.base.field_name)?;
        if !had {
            return Ok(None);
        }
        // finish(): inside.andNot(outside)
        let (Some(mut inside), Some(outside)) = (v.inside, v.outside) else {
            return Ok(None);
        };
        inside.and_not(&outside);
        Ok(Some(inside))
    }
}

/// `bufferShape(shape, distErr)`: the shape grown by `dist_err` -- a circle
/// for a point or circle, else the bounding box grown (clamped to the
/// poles, all longitudes once it reaches one, or to a planar world).
///
/// # Errors
/// `dist_err <= 0`, or a geometry error.
pub fn buffer_shape(
    ctx: &Arc<SpatialContext>,
    shape: &Arc<dyn Shape>,
    dist_err: f64,
) -> Result<Arc<dyn Shape>> {
    if dist_err <= 0.0 {
        return Err(Error::IllegalArgument("distErr must be > 0".into()));
    }
    if let Some(p) = shape.as_point() {
        let center = ctx.point_xy(p.x(), p.y())?;
        return Ok(ctx.circle_at(&center, dist_err)?);
    }
    if let Some(circle) = shape.as_circle() {
        let mut new_dist = circle.radius() + dist_err;
        if ctx.is_geo() && new_dist > 180.0 {
            new_dist = 180.0;
        }
        return Ok(ctx.circle_at(&circle.center()?, new_dist)?);
    }
    let bbox = shape.bounding_box()?;
    let mut new_min_x = bbox.min_x() - dist_err;
    let mut new_max_x = bbox.max_x() + dist_err;
    let mut new_min_y = bbox.min_y() - dist_err;
    let mut new_max_y = bbox.max_y() + dist_err;
    if ctx.is_geo() {
        if new_min_y < -90.0 {
            new_min_y = -90.0;
        }
        if new_max_y > 90.0 {
            new_max_y = 90.0;
        }
        #[allow(clippy::float_cmp)] // the clamped values, exactly
        if new_min_y == -90.0 || new_max_y == 90.0 || bbox.width() + 2.0 * dist_err > 360.0 {
            new_min_x = -180.0;
            new_max_x = 180.0;
        } else {
            new_min_x = DistanceUtils::norm_lon_deg(new_min_x);
            new_max_x = DistanceUtils::norm_lon_deg(new_max_x);
        }
    } else {
        // restrict to world bounds
        let [min_x, max_x, min_y, max_y] = ctx.world_bounds_values();
        new_min_x = java_max(new_min_x, min_x);
        new_max_x = java_min(new_max_x, max_x);
        new_min_y = java_max(new_min_y, min_y);
        new_max_y = java_min(new_max_y, max_y);
    }
    Ok(ctx.rect(new_min_x, new_max_x, new_min_y, new_max_y)?)
}

/// `Math.max` (NaN-propagating, `-0 < +0`).
pub(crate) fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { b } else { a };
    }
    if a >= b {
        a
    } else {
        b
    }
}

/// `Math.min` (NaN-propagating, `-0 < +0`).
pub(crate) fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { a } else { b };
    }
    if a <= b {
        a
    } else {
        b
    }
}

struct WithinVisitor {
    inside: Option<FixedBitSet>,
    outside: Option<FixedBitSet>,
    buffered: Option<Arc<dyn Shape>>,
}

impl WithinVisitor {
    /// `allCellsIntersectQuery(cell)`: whether the cell and every cell
    /// beneath it to the detail level intersect the query shape.
    fn all_cells_intersect_query(q: &VisitingQuery, cell: &dyn Cell) -> Result<bool> {
        let relate = cell.relate_shape(&*q.base.query_shape)?;
        if cell.level() == q.base.detail_level {
            return Ok(relate.intersects());
        }
        if relate == SpatialRelation::Within {
            return Ok(true);
        }
        if relate == SpatialRelation::Disjoint {
            return Ok(false);
        }
        // Note: Generating all these cells just to determine intersection is
        // not ideal (LUCENE-4869).
        let mut sub_cells = cell.next_level_cells(None)?;
        while sub_cells.has_next()? {
            let sub_cell = sub_cells.next()?;
            if !Self::all_cells_intersect_query(q, &*sub_cell)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Visitor for WithinVisitor {
    fn start(&mut self, t: &mut Traverser<'_>) -> Result<()> {
        self.inside = Some(FixedBitSet::new(idx(t.max_doc)));
        self.outside = Some(FixedBitSet::new(idx(t.max_doc)));
        Ok(())
    }

    /// The buffered query shape instead of the original (works with `None`
    /// too: every cell).
    fn find_sub_cells_to_visit(
        &mut self,
        _q: &VisitingQuery,
        cell: &dyn Cell,
    ) -> Result<Box<dyn CellIterator>> {
        Ok(cell.next_level_cells(self.buffered.as_ref())?)
    }

    fn visit_prefix(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<bool> {
        // cell.relate is based on the bufferedQueryShape; we need to examine
        // what the relation is against the queryShape
        let visit_relation = cell.relate_shape(&*q.base.query_shape)?;
        let (inside, outside) = (
            self.inside.as_mut().expect("started"),
            self.outside.as_mut().expect("started"),
        );
        if cell.level() == q.base.detail_level {
            t.collect_docs(if visit_relation.intersects() {
                inside
            } else {
                outside
            })?;
            return Ok(false);
        } else if visit_relation == SpatialRelation::Within {
            t.collect_docs(inside)?;
            return Ok(false);
        } else if visit_relation == SpatialRelation::Disjoint {
            t.collect_docs(outside)?;
            return Ok(false);
        }
        Ok(true)
    }

    fn visit_leaf(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<()> {
        if Self::all_cells_intersect_query(q, cell)? {
            t.collect_docs(self.inside.as_mut().expect("started"))
        } else {
            t.collect_docs(self.outside.as_mut().expect("started"))
        }
    }

    fn scans_by_default(&self) -> bool {
        false
    }

    /// Collects as wanted even when the cell is not a leaf.
    fn visit_scanned(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &mut dyn Cell,
    ) -> Result<()> {
        self.visit_leaf(q, t, cell)
    }
}

impl DocumentQuery for WithinPrefixTreeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        collect_set(leaf, self.doc_id_set(leaf)?.as_ref(), boost, collector);
        Ok(())
    }
}

/// `ContainsPrefixTreeQuery`: documents whose shape contains the query
/// shape -- every query cell at the detail level must be covered by one of
/// the document's cells (an AND over a cell's children of the documents
/// found beneath each).
#[derive(Clone, Debug)]
pub struct ContainsPrefixTreeQuery {
    pub base: PrefixTreeQueryBase,
    /// `multiOverlappingIndexedShapes` (LUCENE-5062): whether a document's
    /// shapes may overlap, so a leaf cannot end the search beneath it.
    pub multi_overlapping_indexed_shapes: bool,
}

/// `SmallDocSet`: a set of doc ids, sorted.
type SmallDocSet = Vec<i32>;

/// `union(aSet, bSet)`.
fn union(a: Option<SmallDocSet>, b: Option<SmallDocSet>) -> Option<SmallDocSet> {
    match (a, b) {
        (a, None) => a,
        (None, b) => b,
        (Some(a), Some(b)) => {
            let mut out = Vec::with_capacity(a.len() + b.len());
            let (mut i, mut j) = (0, 0);
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    std::cmp::Ordering::Less => {
                        out.push(a[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        out.push(b[j]);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        out.push(a[i]);
                        i += 1;
                        j += 1;
                    }
                }
            }
            out.extend_from_slice(&a[i..]);
            out.extend_from_slice(&b[j..]);
            Some(out)
        }
    }
}

struct ContainsVisitor<'q, 'a> {
    q: &'q ContainsPrefixTreeQuery,
    t: Traverser<'a>,
    /// `seekTerm`: one buffer for every seek target.
    seek_term: Vec<u8>,
}

impl ContainsVisitor<'_, '_> {
    /// `visit(cell, acceptContains)`: the primary algorithm, recursive;
    /// `None` when nothing is found.
    fn visit(
        &mut self,
        cell: &dyn Cell,
        accept_contains: Option<&SmallDocSet>,
    ) -> Result<Option<SmallDocSet>> {
        if !self.t.on_term {
            return Ok(None); // signals all done
        }
        // Get the AND of all child results
        let mut combined: Option<SmallDocSet> = None;
        let mut accept: Option<SmallDocSet> = accept_contains.cloned();
        // Optimization: no filter when the cell is within the query shape.
        let mut sub_cells_filter = Some(&self.q.base.query_shape);
        if cell.level() != 0
            && (cell.shape_rel().is_none() || cell.shape_rel() == Some(SpatialRelation::Within))
        {
            sub_cells_filter = None;
        }
        let mut sub_cells = cell.next_level_cells(sub_cells_filter)?;
        while sub_cells.has_next()? {
            let sub_cell = sub_cells.next()?;
            combined = if !self.seek(&*sub_cell)? {
                None
            } else if sub_cell.level() == self.q.base.detail_level {
                self.get_docs(&*sub_cell, accept.as_ref())?
            } else if !self.q.multi_overlapping_indexed_shapes
                && sub_cell.shape_rel() == Some(SpatialRelation::Within)
            {
                self.get_leaf_docs(&*sub_cell, accept.as_ref())?
            } else {
                // OR the leaf docs with all child results
                let leaf_docs = self.get_leaf_docs(&*sub_cell, accept.as_ref())?;
                let sub_docs = self.visit(&*sub_cell, accept.as_ref())?; // recursion
                union(leaf_docs, sub_docs)
            };
            match &combined {
                None => break,
                // has the 'AND' effect on next iteration
                Some(c) => accept = Some(c.clone()),
            }
        }
        Ok(combined)
    }

    /// `seek(cell)`: whether the terms are now on `cell` (a leaf of it
    /// included).
    fn seek(&mut self, cell: &dyn Cell) -> Result<bool> {
        if !self.t.on_term {
            return Ok(false);
        }
        let compare = self.t.cell().compare_to_no_leaf(cell);
        if compare > 0 {
            return Ok(false); // leap-frog effect
        } else if compare == 0 {
            return Ok(true); // already there!
        }
        // seek!
        cell.token_bytes_no_leaf_into(&mut self.seek_term);
        let status = self.t.try_seek_ceil(&self.seek_term)?;
        if status == SeekStatus::End {
            return Ok(false); // all done (on_term is false)
        }
        if status == SeekStatus::Found {
            return Ok(true);
        }
        Ok(self.t.cell().is_leaf() && self.t.cell().compare_to_no_leaf(cell) == 0)
    }

    /// `getDocs(cell, acceptContains)`: the prefix's and the leaf's docs at
    /// the detail level.
    fn get_docs(
        &mut self,
        cell: &dyn Cell,
        accept: Option<&SmallDocSet>,
    ) -> Result<Option<SmallDocSet>> {
        if self.t.cell().is_leaf() {
            // only a leaf
            let result = self.collect_docs(accept)?;
            self.t.next_term()?;
            return Ok(result);
        }
        let docs_at_prefix = self.collect_docs(accept)?;
        if !self.t.next_term()? {
            return Ok(docs_at_prefix);
        }
        // collect leaf too
        if self.t.cell().is_leaf() && self.t.cell().compare_to_no_leaf(cell) == 0 {
            let docs_at_leaf = self.collect_docs(accept)?;
            self.t.next_term()?;
            return Ok(union(docs_at_prefix, docs_at_leaf));
        }
        Ok(docs_at_prefix)
    }

    /// `getLeafDocs(cell, acceptContains)`: the docs of the cell's leaf, if
    /// it has one.
    fn get_leaf_docs(
        &mut self,
        cell: &dyn Cell,
        accept: Option<&SmallDocSet>,
    ) -> Result<Option<SmallDocSet>> {
        // Advance past prefix if we're at a prefix; None if no leaf
        if !self.t.cell().is_leaf()
            && (!self.t.next_term()?
                || !self.t.cell().is_leaf()
                || self.t.cell().level() != cell.level())
        {
            return Ok(None);
        }
        let result = self.collect_docs(accept)?;
        self.t.next_term()?;
        Ok(result)
    }

    /// `collectDocs(acceptContains)`: `None` for no documents.
    fn collect_docs(&mut self, accept: Option<&SmallDocSet>) -> Result<Option<SmallDocSet>> {
        let mut set: SmallDocSet = Vec::new();
        self.t.for_each_doc(|doc| {
            if accept.is_none_or(|a| a.binary_search(&doc).is_ok()) {
                set.push(doc);
            }
            true
        })?;
        set.sort_unstable();
        set.dedup();
        Ok(if set.is_empty() { None } else { Some(set) })
    }
}

impl ContainsPrefixTreeQuery {
    pub fn new(
        query_shape: Arc<dyn Shape>,
        field_name: &str,
        grid: Arc<dyn SpatialPrefixTree>,
        detail_level: i32,
        multi_overlapping_indexed_shapes: bool,
    ) -> Self {
        ContainsPrefixTreeQuery {
            base: PrefixTreeQueryBase {
                query_shape,
                field_name: field_name.to_string(),
                grid,
                detail_level,
            },
            multi_overlapping_indexed_shapes,
        }
    }

    fn docs(&self, leaf: &OpenSegment<'_>) -> Result<Option<SmallDocSet>> {
        let mut t = Traverser::new(leaf, &self.base.field_name, &*self.base.grid)?;
        if t.has_terms() {
            t.next_term()?; // advance to first
        }
        let mut v = ContainsVisitor {
            q: self,
            t,
            seek_term: Vec::new(),
        };
        let world = self.base.grid.world_cell();
        v.visit(&*world, None)
    }
}

impl DocumentQuery for ContainsPrefixTreeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let max_doc = reader(leaf)?.max_doc;
        if let Some(docs) = self.docs(leaf)? {
            if let Some(&bad) = docs.iter().find(|&&d| d < 0 || d >= max_doc) {
                return check_walk(Some(bad), &self.base.field_name, max_doc);
            }
            for doc in docs {
                collect_live(leaf, doc, boost, collector);
            }
        }
        Ok(())
    }
}

/// The `TermQuery` RPT makes of a point on a points-only field: the point's
/// leaf cell, scored by BM25 as Java's `TermQuery` is -- the term's
/// document frequency and the field's statistics summed over the segments
/// (deleted documents included), `freq` 1 and no norms (the field is
/// `DOCS`, norms omitted).
#[derive(Clone, Debug)]
pub struct PrefixTreeTermQuery {
    pub field: String,
    pub term: Vec<u8>,
    /// `(idf, avgdl)`, once rewritten against the reader.
    stats: Option<(f32, f32)>,
}

impl PrefixTreeTermQuery {
    pub fn new(field: &str, term: Vec<u8>) -> Self {
        PrefixTreeTermQuery {
            field: field.to_string(),
            term,
            stats: None,
        }
    }
}

impl DocumentQuery for PrefixTreeTermQuery {
    /// `createWeight`'s `TermStates.build` and collection statistics: the
    /// sums over `leaves`.
    fn rewrite(&self, leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        if self.stats.is_some() {
            return Ok(None);
        }
        let (mut doc_freq, mut doc_count, mut sum_ttf) = (0i64, 0i64, 0i64);
        for leaf in leaves {
            if let Some(ft) = leaf.fields.field(&self.field) {
                doc_count += i64::from(ft.doc_count);
                sum_ttf += ft.sum_total_term_freq;
                if let Some(seeked) = ft.seek_term_state(&self.term)? {
                    doc_freq += i64::from(seeked.stats.doc_freq);
                }
            }
        }
        let idf = if doc_freq == 0 {
            0.0
        } else {
            crate::similarity::idf(doc_freq, doc_count)
        };
        let avgdl = if doc_count == 0 {
            1.0
        } else {
            (sum_ttf as f64 / doc_count as f64) as f32
        };
        Ok(Some(Box::new(PrefixTreeTermQuery {
            field: self.field.clone(),
            term: self.term.clone(),
            stats: Some((idf, avgdl)),
        })))
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let (idf, avgdl) = self.stats.unwrap_or((0.0, 1.0));
        let Some(ft) = leaf.fields.field(&self.field) else {
            return Ok(());
        };
        let Some(postings) = ft.postings_with_flags(
            &self.term,
            leaf.doc_in,
            lucene_codecs::postings::PostingsFlags::DocsOnly,
        )?
        else {
            return Ok(());
        };
        let weight = boost * idf;
        let norm_inverse = crate::similarity::norm_inverse(
            1.0,
            avgdl,
            crate::similarity::DEFAULT_K1,
            crate::similarity::DEFAULT_B,
        );
        let score = crate::similarity::do_score(weight, 1.0, norm_inverse);
        let max_doc = reader(leaf)?.max_doc;
        for doc in postings.docs {
            if doc < 0 || doc >= max_doc {
                return check_walk(Some(doc), &self.field, max_doc);
            }
            collect_live(leaf, doc, score, collector);
        }
        Ok(())
    }
}

/// The `TermInSetQuery` [`super::TermQueryPrefixTreeStrategy`] makes: the
/// live documents holding any of the terms (from the postings), at a
/// constant score.
#[derive(Clone, Debug)]
pub struct PrefixTreeTermsQuery {
    pub field: String,
    /// Sorted, deduplicated.
    pub terms: Vec<Vec<u8>>,
}

impl DocumentQuery for PrefixTreeTermsQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(ft) = leaf.fields.field(&self.field) else {
            return Ok(());
        };
        let max_doc = reader(leaf)?.max_doc;
        let mut bits = FixedBitSet::new(idx(max_doc));
        let mut bad = None;
        for term in &self.terms {
            if let Some(p) = ft.postings_with_flags(
                term,
                leaf.doc_in,
                lucene_codecs::postings::PostingsFlags::DocsOnly,
            )? {
                for doc in p.docs {
                    set_doc(&mut bits, doc, &mut bad);
                }
            }
        }
        check_walk(bad, &self.field, max_doc)?;
        collect_bits(leaf, &bits, boost, collector);
        Ok(())
    }
}
