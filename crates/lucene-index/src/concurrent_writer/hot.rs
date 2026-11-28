//! The hot tier ([`crate::hot_tier`]) on a [`ConcurrentIndexWriter`]: turning
//! it on, applying what it holds, and writing what stands for the updates it
//! drops.

use std::sync::Arc;

use super::{lock, ConcurrentIndexWriter, DeleteNode, SeqNo};
use crate::buffered_updates::DocValuesUpdate;
use crate::index_writer::{ExplicitDocument, Result};

impl<'d> ConcurrentIndexWriter<'d> {
    /// Turns the hot tier on, once: see [`crate::hot_tier`].
    pub fn enable_hot_tier(&self, cfg: crate::hot_tier::HotTierConfig) -> bool {
        self.hot_cfg.set(cfg).is_ok()
    }

    /// The tier, built by the first update that needs it.
    pub(super) fn hot_tier(&self) -> Option<&Arc<crate::hot_tier::HotTier>> {
        let cfg = self.hot_cfg.get()?;
        Some(
            self.hot
                .get_or_init(|| Arc::new(crate::hot_tier::HotTier::new(cfg.clone()))),
        )
    }

    /// The hot tier's counters, when it is on.
    pub fn hot_tier_stats(&self) -> Option<crate::hot_tier::HotTierStats> {
        self.hot_cfg
            .get()
            .map(|_| self.hot.get().map(|h| h.stats()).unwrap_or_default())
    }

    /// Applies every held operation: what a flush, a commit, a snapshot and
    /// a live view do first. Returns how many.
    pub fn drain_hot_tier(&self) -> Result<usize> {
        match self.hot.get() {
            Some(hot) => {
                self.hot_check()?;
                hot.drain(&mut |op| self.apply_held(op).map(|_| ()))
                    .inspect_err(|e| self.hot_fail(e))
            }
            None => Ok(0),
        }
    }

    /// Applies the held update of every row `docs` adds, before they are
    /// added. A plain add of a row (an index right after a delete: the engine
    /// knows the row has no live version, so it adds rather than updates)
    /// follows an update of that row the tier still holds; applied later, that
    /// update's delete of the row's documents would reach the added one.
    pub(super) fn settle_held_rows(&self, docs: &[ExplicitDocument]) -> Result<()> {
        let Some(hot) = self.hot.get().filter(|h| !h.is_empty()) else {
            return Ok(());
        };
        let Some(field) = hot.term_field() else {
            return Ok(());
        };
        let Some(number) = self
            .cfg()
            .field_list()
            .iter()
            .find(|f| f.name == field)
            .map(|f| f.number)
        else {
            return Ok(());
        };
        self.hot_check()?;
        for f in docs.iter().flat_map(|d| &d.fields.inverted) {
            if f.field_number != number {
                continue;
            }
            for term in f.terms.iter() {
                hot.settle(term.term, &mut |op| self.apply_held(op).map(|_| ()))
                    .inspect_err(|e| self.hot_fail(e))?;
            }
        }
        Ok(())
    }

    /// Fails once applying a held operation has failed.
    pub(super) fn hot_check(&self) -> Result<()> {
        match lock(&self.hot_failed).as_ref() {
            Some(why) => Err(crate::index_writer::explicit_error(format!(
                "a held update could not be applied: {why}"
            ))),
            None => Ok(()),
        }
    }

    pub(super) fn hot_fail(&self, e: &crate::index_writer::Error) {
        lock(&self.hot_failed).get_or_insert_with(|| e.to_string());
    }

    /// Writes what stands for `dropped`, an operation the tier replaced: a
    /// small no-op that keeps its sequence number in the index's history, or
    /// the whole operation when no such document can be built.
    pub(super) fn keep_history(
        &self,
        hot: &crate::hot_tier::HotTier,
        dropped: Option<crate::hot_tier::HeldOp>,
    ) -> Result<()> {
        let Some(dropped) = dropped else {
            return Ok(());
        };
        let Some(stub) = hot.history() else {
            return Ok(());
        };
        match self.history_stub(stub, &dropped) {
            Some(doc) => self
                .add_explicit(dropped.partition.clone(), None, vec![doc], None)
                .map(|_| ()),
            None => self.apply_held(dropped).map(|_| ()),
        }
    }

    /// The no-op document for `dropped`: its sequence number and primary term,
    /// the constants, the soft-deletes mark; `None` when a field is missing.
    pub(super) fn history_stub(
        &self,
        stub: &crate::hot_tier::HistoryStub,
        dropped: &crate::hot_tier::HeldOp,
    ) -> Option<ExplicitDocument> {
        let cfg = self.cfg();
        let number = |name: &str| {
            cfg.field_list()
                .iter()
                .find(|f| f.name == name)
                .map(|f| f.number)
        };
        let copy: Vec<i32> = stub.copy.iter().map(|n| number(n)).collect::<Option<_>>()?;
        let mut constants: Vec<(i32, i64)> = stub
            .constants
            .iter()
            .map(|(n, v)| number(n).map(|n| (n, *v)))
            .collect::<Option<_>>()?;
        for u in &dropped.soft_deletes {
            if let DocValuesUpdate::Numeric { field, .. } = u {
                constants.push((number(field)?, 1));
            }
        }
        let carrier = dropped.docs.iter().rev().find(|d| {
            d.fields
                .doc_values
                .iter()
                .any(|f| copy.contains(&f.field_number))
        })?;
        let mut doc = ExplicitDocument::default();
        let keep = |f: &&lucene_codecs::stored_fields::StoredField| copy.contains(&f.field_number);
        doc.fields.doc_values = carrier
            .fields
            .doc_values
            .iter()
            .filter(keep)
            .cloned()
            .collect();
        doc.fields.points = carrier.fields.points.iter().filter(keep).cloned().collect();
        for (n, v) in constants {
            doc.fields
                .doc_values
                .push(lucene_codecs::stored_fields::StoredField {
                    field_number: n,
                    value: lucene_codecs::stored_fields::FieldValue::Long(v),
                });
        }
        if let Some((name, value)) = &stub.source {
            doc.stored.push(lucene_codecs::stored_fields::StoredField {
                field_number: number(name)?,
                value: lucene_codecs::stored_fields::FieldValue::Binary(value.clone()),
            });
        }
        Some(doc)
    }

    pub(super) fn apply_held(&self, op: crate::hot_tier::HeldOp) -> Result<SeqNo> {
        self.add_explicit(
            op.partition,
            Some(DeleteNode::DocValuesUpdates(op.soft_deletes)),
            op.docs,
            op.vectors,
        )
    }
}
