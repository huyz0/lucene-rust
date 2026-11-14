//! The version of the contract between this library and the plugin.

/// The version of the contract between this library and the Java classes in
/// `opensearch-plugin/`: every exported entry point, its argument order, and the
/// query blob layout. The plugin refuses to
/// start against a library reporting any other number -- a jar carrying a
/// stale `.so` must not get as far as reading an index.
///
/// Bump it on any change a Java caller could observe.
pub const JVM_ABI_VERSION: u32 = 180;

/// [`JVM_ABI_VERSION`], for the plugin's load-time handshake.
#[no_mangle]
pub extern "C" fn ffi_jvm_abi_version() -> u32 {
    JVM_ABI_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_the_constant() {
        assert_eq!(ffi_jvm_abi_version(), JVM_ABI_VERSION);
    }
}
