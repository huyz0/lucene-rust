//! `BaseLockFactoryTestCase` over every lock factory, as Lucene's
//! `TestNativeFSLockFactory`, `TestSimpleFSLockFactory`,
//! `TestSingleInstanceLockFactory` and `TestSleepingLockWrapper` run it.

use std::path::Path;
use std::sync::Arc;

use lucene_store::{
    ByteBuffersDirectory, FsDirectory, NativeFsLockFactory, SimpleFsLockFactory,
    SingleInstanceLockFactory, SleepingLockWrapper,
};
use lucene_test_framework::base_lock_factory_tests;
use lucene_test_framework::store::base_lock_factory::BaseLockFactoryTestCase;

mod native_fs {
    use super::*;
    base_lock_factory_tests!(BaseLockFactoryTestCase::new(|p: &Path| {
        FsDirectory::with_lock_factory(p, Arc::new(NativeFsLockFactory::INSTANCE))
    }));
}

mod simple_fs {
    use super::*;
    base_lock_factory_tests!(BaseLockFactoryTestCase::new(|p: &Path| {
        FsDirectory::with_lock_factory(p, Arc::new(SimpleFsLockFactory::INSTANCE))
    }));
}

mod single_instance {
    use super::*;
    base_lock_factory_tests!(BaseLockFactoryTestCase::new(|_: &Path| {
        ByteBuffersDirectory::with_lock_factory(Arc::new(SingleInstanceLockFactory::new()))
    }));
}

/// `TestSleepingLockWrapper.getDirectory`: a short wait over a
/// single-instance lock factory (a failed obtain still fails, after
/// waiting).
mod sleeping {
    use super::*;
    base_lock_factory_tests!(BaseLockFactoryTestCase::new(|_: &Path| {
        SleepingLockWrapper::with_poll_interval(
            ByteBuffersDirectory::with_lock_factory(Arc::new(SingleInstanceLockFactory::new())),
            20,
            2,
        )
        .unwrap()
    }));
}
