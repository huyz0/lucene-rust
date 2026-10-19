//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::basic_index;

#[test]
fn a_manager_over_commits_refreshes_and_closes() {
    let idx = basic_index();
    let m = SearcherTaxonomyManager::from_directories(&idx.index, &idx.taxo).unwrap();
    assert_eq!(
        format!("{m:?}"),
        "SearcherTaxonomyManager { taxo_epoch: -1, .. }"
    );
    m.maybe_refresh_blocking().unwrap();
    {
        // Another refresh running: this one does not wait.
        let _held = m.refresh_lock.lock().unwrap();
        assert!(!m.maybe_refresh().unwrap());
    }
    assert!(m.maybe_refresh().unwrap());
    let r = m.acquire().unwrap();
    assert_eq!(r.searcher.max_doc(), 5);
    m.release(r);
    m.close();
    assert!(matches!(m.acquire(), Err(Error::AlreadyClosed(_))));
    assert!(matches!(m.maybe_refresh(), Err(Error::AlreadyClosed(_))));
    assert!(matches!(
        m.maybe_refresh_blocking(),
        Err(Error::AlreadyClosed(_))
    ));
}
