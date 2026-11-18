//! The version of the contract between this library and the plugin.

/// The version of the contract between this library and the Java classes in
/// `opensearch-plugin/`: every exported entry point, its argument order, and the
/// query blob layout. The plugin refuses to
/// start against a library reporting any other number -- a jar carrying a
/// stale `.so` must not get as far as reading an index.
///
/// Bump it on any change a Java caller could observe.
pub const JVM_ABI_VERSION: u32 = 198;

/// [`JVM_ABI_VERSION`], for the plugin's load-time handshake.
#[no_mangle]
pub extern "C" fn ffi_jvm_abi_version() -> u32 {
    JVM_ABI_VERSION
}

/// Tells the library which Lucene release the host node runs (the plugin
/// passes `org.apache.lucene.util.Version.LATEST` once at start), so that
/// analysis, scoring and join behaviour that differs between releases is the
/// node's own: OpenSearch 3.9 bundles Lucene 10.5.1, 3.11 bundles 10.6.0. A
/// release newer than the library knows counts as the newest it knows.
#[no_mangle]
pub extern "C" fn ffi_set_host_lucene_version(major: i32, minor: i32, bugfix: i32) -> i32 {
    use crate::error::{guard, set_last_error, FfiStatus};
    guard(|| {
        let host = lucene_util::version::Version::from_bits(major, minor, bugfix).map_err(|e| {
            set_last_error(format!("ffi_set_host_lucene_version: {e}"));
            FfiStatus::InvalidArgument
        })?;
        lucene_util::version::Version::set_host(host);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_the_constant() {
        assert_eq!(ffi_jvm_abi_version(), JVM_ABI_VERSION);
    }

    #[test]
    fn host_lucene_version_is_validated_and_recorded() {
        use lucene_util::version::Version;
        assert_eq!(
            ffi_set_host_lucene_version(300, 0, 0),
            10,
            "InvalidArgument"
        );
        assert_eq!(ffi_set_host_lucene_version(10, 6, 0), 0);
        assert_eq!(Version::host(), Version::LUCENE_10_6_0);
    }
}
