//! Port of `org.apache.lucene.index.FilterMergePolicy` (delegate every
//! decision to a wrapped policy, overriding what a subclass needs) and
//! `OneMergeWrappingMergePolicy` (rewrite each [`OneMerge`] the delegate
//! proposes).
//!
//! Java subclasses `FilterMergePolicy` and overrides methods; the Rust shape
//! is the same delegation as a struct holding the inner `Box<dyn
//! MergePolicy>`, reachable through [`FilterMergePolicy::unwrap`]/
//! [`FilterMergePolicy::inner_mut`] (`Unwrappable.unwrap`). A policy that
//! needs to override only a few methods implements [`MergePolicy`] itself and
//! forwards the rest to a `FilterMergePolicy` field.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use super::api::{
    CompoundFileSettings, MergeContext, MergePolicy, MergeSegment, MergeSpecification,
    MergeTrigger, OneMerge, Result,
};

/// `FilterMergePolicy`: every method answered by the wrapped policy.
#[derive(Debug)]
pub struct FilterMergePolicy {
    inner: Box<dyn MergePolicy>,
}

impl FilterMergePolicy {
    pub fn new(inner: Box<dyn MergePolicy>) -> Self {
        FilterMergePolicy { inner }
    }

    /// `Unwrappable.unwrap`.
    pub fn unwrap(&self) -> &dyn MergePolicy {
        self.inner.as_ref()
    }

    /// The wrapped policy, mutably (for its own setters).
    pub fn inner_mut(&mut self) -> &mut dyn MergePolicy {
        self.inner.as_mut()
    }

    /// The wrapped policy, by value.
    pub fn into_inner(self) -> Box<dyn MergePolicy> {
        self.inner
    }
}

impl MergePolicy for FilterMergePolicy {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.inner.find_merges(trigger, infos, ctx)
    }
    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.inner
            .find_forced_merges(infos, max_segment_count, segments_to_merge, ctx)
    }
    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.inner.find_forced_deletes_merges(infos, ctx)
    }
    fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
        self.inner.find_merges_for_readers(max_docs)
    }
    fn find_full_flush_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        self.inner.find_full_flush_merges(trigger, infos, ctx)
    }
    fn use_compound_file(
        &self,
        infos: &[MergeSegment],
        merged: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        self.inner.use_compound_file(infos, merged, ctx)
    }
    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        self.inner.size(info, ctx)
    }
    fn max_full_flush_merge_size(&self) -> i64 {
        self.inner.max_full_flush_merge_size()
    }
    fn keep_fully_deleted_segment(&self) -> bool {
        self.inner.keep_fully_deleted_segment()
    }
    fn num_deletes_to_merge(&self, info: &MergeSegment, del_count: i32) -> i32 {
        self.inner.num_deletes_to_merge(info, del_count)
    }
    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.inner.compound_file_settings()
    }
    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        self.inner.compound_file_settings_mut()
    }
}

/// The `UnaryOperator<OneMerge>` a [`OneMergeWrappingMergePolicy`] applies.
pub type OneMergeWrapper = Arc<dyn Fn(OneMerge) -> OneMerge + Send + Sync>;

/// `OneMergeWrappingMergePolicy`: the delegate's natural, forced,
/// forced-deletes and full-flush specifications, each merge passed through
/// `wrap`. Java's wrappers typically override `OneMerge.wrapForMerge`/
/// `reorder`: here `wrap` returns the merge with its hooks
/// ([`OneMerge::with_hooks`]), or rewrites the merge itself.
pub struct OneMergeWrappingMergePolicy {
    filter: FilterMergePolicy,
    wrap: OneMergeWrapper,
}

impl fmt::Debug for OneMergeWrappingMergePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneMergeWrappingMergePolicy")
            .field("in", &self.filter.inner)
            .finish()
    }
}

impl OneMergeWrappingMergePolicy {
    pub fn new(inner: Box<dyn MergePolicy>, wrap: OneMergeWrapper) -> Self {
        OneMergeWrappingMergePolicy {
            filter: FilterMergePolicy::new(inner),
            wrap,
        }
    }

    /// `Unwrappable.unwrap`.
    pub fn unwrap(&self) -> &dyn MergePolicy {
        self.filter.unwrap()
    }

    /// `wrapSpec`.
    fn wrap_spec(&self, spec: Option<MergeSpecification>) -> Option<MergeSpecification> {
        spec.map(|spec| MergeSpecification {
            merges: spec.merges.into_iter().map(|m| (self.wrap)(m)).collect(),
        })
    }
}

impl MergePolicy for OneMergeWrappingMergePolicy {
    fn find_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(self.wrap_spec(self.filter.find_merges(trigger, infos, ctx)?))
    }
    fn find_forced_merges(
        &self,
        infos: &[MergeSegment],
        max_segment_count: i32,
        segments_to_merge: &HashMap<String, bool>,
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(self.wrap_spec(self.filter.find_forced_merges(
            infos,
            max_segment_count,
            segments_to_merge,
            ctx,
        )?))
    }
    fn find_forced_deletes_merges(
        &self,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(self.wrap_spec(self.filter.find_forced_deletes_merges(infos, ctx)?))
    }
    fn find_merges_for_readers(&self, max_docs: &[i64]) -> Option<MergeSpecification> {
        // Not overridden in Java: `FilterMergePolicy.findMerges(CodecReader...)`.
        self.filter.find_merges_for_readers(max_docs)
    }
    fn find_full_flush_merges(
        &self,
        trigger: MergeTrigger,
        infos: &[MergeSegment],
        ctx: &dyn MergeContext,
    ) -> Result<Option<MergeSpecification>> {
        Ok(self.wrap_spec(self.filter.find_full_flush_merges(trigger, infos, ctx)?))
    }
    fn use_compound_file(
        &self,
        infos: &[MergeSegment],
        merged: &MergeSegment,
        ctx: &dyn MergeContext,
    ) -> Result<bool> {
        self.filter.use_compound_file(infos, merged, ctx)
    }
    fn size(&self, info: &MergeSegment, ctx: &dyn MergeContext) -> Result<i64> {
        self.filter.size(info, ctx)
    }
    fn max_full_flush_merge_size(&self) -> i64 {
        self.filter.max_full_flush_merge_size()
    }
    fn keep_fully_deleted_segment(&self) -> bool {
        self.filter.keep_fully_deleted_segment()
    }
    fn num_deletes_to_merge(&self, info: &MergeSegment, del_count: i32) -> i32 {
        self.filter.num_deletes_to_merge(info, del_count)
    }
    fn compound_file_settings(&self) -> CompoundFileSettings {
        self.filter.compound_file_settings()
    }
    fn compound_file_settings_mut(&mut self) -> &mut CompoundFileSettings {
        self.filter.compound_file_settings_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::super::api::{BasicMergeContext, TieredMergePolicy};
    use super::super::log::LogDocMergePolicy;
    use super::*;

    fn infos() -> Vec<MergeSegment> {
        (0..4)
            .map(|i| MergeSegment::new(format!("_{i}"), 10, i, 100))
            .collect()
    }

    #[test]
    fn filter_delegates_everything() {
        let ctx = BasicMergeContext::default();
        let mut inner = LogDocMergePolicy::new();
        inner.set_merge_factor(2).unwrap();
        inner.set_min_merge_docs(1);
        let expected = inner
            .find_merges(MergeTrigger::Explicit, &infos(), &ctx)
            .unwrap()
            .map(|s| s.groups());
        let mut f = FilterMergePolicy::new(Box::new(inner.clone()));
        let t = MergeTrigger::Explicit;
        assert_eq!(
            f.find_merges(t, &infos(), &ctx)
                .unwrap()
                .map(|s| s.groups()),
            expected
        );
        let all: HashMap<String, bool> = infos().iter().map(|s| (s.name.clone(), true)).collect();
        assert_eq!(
            f.find_forced_merges(&infos(), 1, &all, &ctx)
                .unwrap()
                .map(|s| s.groups()),
            inner
                .find_forced_merges(&infos(), 1, &all, &ctx)
                .unwrap()
                .map(|s| s.groups())
        );
        assert_eq!(
            f.find_forced_deletes_merges(&infos(), &ctx)
                .unwrap()
                .map(|s| s.groups()),
            inner
                .find_forced_deletes_merges(&infos(), &ctx)
                .unwrap()
                .map(|s| s.groups())
        );
        assert_eq!(
            f.find_full_flush_merges(t, &infos(), &ctx)
                .unwrap()
                .map(|s| s.groups()),
            inner
                .find_full_flush_merges(t, &infos(), &ctx)
                .unwrap()
                .map(|s| s.groups())
        );
        let i = infos();
        assert_eq!(
            f.use_compound_file(&i, &i[0], &ctx).unwrap(),
            inner.use_compound_file(&i, &i[0], &ctx).unwrap()
        );
        assert_eq!(f.size(&i[1], &ctx).unwrap(), 9);
        assert_eq!(f.max_full_flush_merge_size(), 1);
        assert!(!f.keep_fully_deleted_segment());
        assert_eq!(f.num_deletes_to_merge(&i[0], 4), 4);
        assert_eq!(f.find_merges_for_readers(&[1]).unwrap().merges.len(), 1);
        f.compound_file_settings_mut()
            .set_no_cfs_ratio(0.7)
            .unwrap();
        assert_eq!(f.unwrap().compound_file_settings().no_cfs_ratio, 0.7);
        assert_eq!(f.compound_file_settings().no_cfs_ratio, 0.7);
        let _ = f.inner_mut();
        assert!(format!("{:?}", f.into_inner()).contains("LogMergePolicy"));
    }

    #[test]
    fn wrapping_policy_rewrites_every_merge() {
        let ctx = BasicMergeContext::default();
        let mut inner = LogDocMergePolicy::new();
        inner.set_merge_factor(2).unwrap();
        inner.set_min_merge_docs(1);
        let wrap: OneMergeWrapper = Arc::new(|mut m: OneMerge| {
            m.segments.reverse();
            m
        });
        let mut p = OneMergeWrappingMergePolicy::new(Box::new(inner), wrap);
        let t = MergeTrigger::Explicit;
        let got = p.find_merges(t, &infos(), &ctx).unwrap().unwrap().groups();
        assert_eq!(got, vec![vec!["_1", "_0"], vec!["_3", "_2"]]);
        let del = p
            .find_forced_deletes_merges(&infos(), &ctx)
            .unwrap()
            .unwrap();
        assert_eq!(del.groups(), vec![vec!["_2", "_1"], vec!["_3"]]);
        let all: HashMap<String, bool> = infos().iter().map(|s| (s.name.clone(), true)).collect();
        let forced = p
            .find_forced_merges(&infos(), 2, &all, &ctx)
            .unwrap()
            .unwrap();
        assert!(forced.groups().iter().all(|g| g.len() >= 2));
        assert!(p
            .find_full_flush_merges(t, &infos(), &ctx)
            .unwrap()
            .is_none());
        let i = infos();
        assert!(!p.use_compound_file(&i, &i[3], &ctx).unwrap());
        assert_eq!(p.size(&i[3], &ctx).unwrap(), 7);
        assert_eq!(p.max_full_flush_merge_size(), 1);
        assert!(!p.keep_fully_deleted_segment());
        assert_eq!(p.num_deletes_to_merge(&i[0], 2), 2);
        assert_eq!(p.find_merges_for_readers(&[1, 2]).unwrap().merges.len(), 1);
        p.compound_file_settings_mut()
            .set_no_cfs_ratio(0.2)
            .unwrap();
        assert_eq!(p.compound_file_settings().no_cfs_ratio, 0.2);
        assert!(format!("{p:?}").contains("OneMergeWrapping"));
        let _ = p.unwrap();
        // `None` stays `None`.
        let tiered = OneMergeWrappingMergePolicy::new(
            Box::new(TieredMergePolicy::default()),
            Arc::new(|m| m),
        );
        assert!(tiered.find_merges(t, &[], &ctx).unwrap().is_none());
    }
}
