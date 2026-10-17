use super::*;
use crate::facets::taxonomy::IntArray;
use std::cell::Cell;

#[derive(Debug)]
struct Flat(Vec<i32>);

impl IntArray for Flat {
    fn get(&self, i: i32) -> i32 {
        self.0[i as usize]
    }
    fn length(&self) -> i32 {
        self.0.len() as i32
    }
}

#[derive(Debug)]
struct Arrays {
    parents: Flat,
    children: Flat,
    siblings: Flat,
}

impl ParallelTaxonomyArrays for Arrays {
    fn parents(&self) -> &dyn IntArray {
        &self.parents
    }
    fn children(&self) -> &dyn IntArray {
        &self.children
    }
    fn siblings(&self) -> &dyn IntArray {
        &self.siblings
    }
}

/// root(0) with a(1) and b(2); `a` has c(3). Its `doClose` can be told to
/// fail.
struct Fake {
    rc: RefCount,
    fail_close: Cell<bool>,
    labels: Vec<FacetLabel>,
}

impl Fake {
    fn new() -> Self {
        Fake {
            rc: RefCount::default(),
            fail_close: Cell::new(false),
            labels: vec![
                FacetLabel::root(),
                FacetLabel::new(["a"]).unwrap(),
                FacetLabel::new(["b"]).unwrap(),
                FacetLabel::new(["a", "c"]).unwrap(),
            ],
        }
    }
}

impl TaxonomyReader for Fake {
    fn ref_counter(&self) -> &RefCount {
        &self.rc
    }
    fn do_close(&self) -> Result<()> {
        if self.fail_close.get() {
            return Err(Error::IllegalState("close failed".into()));
        }
        Ok(())
    }
    fn parallel_taxonomy_arrays(&self) -> Result<Arc<dyn ParallelTaxonomyArrays>> {
        Ok(Arc::new(Arrays {
            parents: Flat(vec![-1, 0, 0, 1]),
            children: Flat(vec![2, 3, -1, -1]),
            siblings: Flat(vec![-1, -1, 1, -1]),
        }))
    }
    fn commit_user_data(&self) -> Result<Vec<(String, String)>> {
        Ok(Vec::new())
    }
    fn get_ordinal(&self, cp: &FacetLabel) -> Result<i32> {
        Ok(self
            .labels
            .iter()
            .position(|l| l == cp)
            .map_or(INVALID_ORDINAL, |p| p as i32))
    }
    fn get_path(&self, ordinal: i32) -> Result<FacetLabel> {
        Ok(self.labels[ordinal as usize].clone())
    }
    fn get_size(&self) -> Result<i32> {
        Ok(self.labels.len() as i32)
    }
}

/// The provided methods: bulk lookups one by one, children youngest first.
#[test]
fn provided_methods() {
    let r = Fake::new();
    assert_eq!(r.get_ordinal_of("a", &["c"]).unwrap(), 3);
    assert!(r.get_ordinal_of("", &["c"]).is_err());
    let labels = [
        FacetLabel::new(["b"]).unwrap(),
        FacetLabel::new(["x"]).unwrap(),
    ];
    assert_eq!(r.get_bulk_ordinals(&labels).unwrap(), [2, -1]);
    assert_eq!(
        r.get_bulk_path(&[3, 0]).unwrap(),
        [r.labels[3].clone(), FacetLabel::root()]
    );
    assert_eq!(r.get_children(0).unwrap().collect::<Vec<_>>(), [2, 1]);
    assert_eq!(r.get_children(-5).unwrap().count(), 0);
    assert!(r.get_children(4).is_err());
    assert!(r.commit_user_data().unwrap().is_empty());
    assert_eq!(r.get_size().unwrap(), 4);
    assert_eq!(ROOT_ORDINAL, 0);
    let it = r.get_children(1).unwrap();
    assert!(format!("{:?}", it.clone()).contains("ChildrenIterator"));
}

/// `decRef` to zero closes; a failing `doClose` restores the count, as
/// Java's `finally` does.
#[test]
fn ref_counting() {
    let r = Fake::new();
    r.fail_close.set(true);
    let err = r.close().unwrap_err();
    assert!(matches!(err, Error::IllegalState(ref m) if m == "close failed"));
    assert_eq!(r.ref_count(), 1);
    r.fail_close.set(false);
    r.close().unwrap();
    assert_eq!(r.ref_count(), 0);
    assert!(r.ensure_open().is_err());
    assert!(!r.try_inc_ref());
    assert!(format!("{:?}", r.rc).contains("RefCount"));
}

/// `incRef`/`tryIncRef` add references that `decRef` drops; only the last
/// one closes, and an `incRef` on a closed reader is refused.
#[test]
fn inc_and_dec_ref() {
    let r = Fake::new();
    r.inc_ref().unwrap();
    assert!(r.try_inc_ref());
    assert_eq!(r.ref_count(), 3);
    r.dec_ref().unwrap();
    r.dec_ref().unwrap();
    assert_eq!(r.ref_count(), 1);
    r.ensure_open().unwrap();
    r.dec_ref().unwrap();
    assert_eq!(r.ref_count(), 0);
    assert!(matches!(r.inc_ref(), Err(Error::AlreadyClosed(_))));
    assert!(matches!(r.dec_ref(), Err(Error::AlreadyClosed(_))));
    // close() after the last decRef has already marked the reader closed.
    r.close().unwrap();
    assert_eq!(r.ref_count(), 0);
}
