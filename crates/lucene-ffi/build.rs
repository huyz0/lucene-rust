//! Links `native/glibc_compat.c` on Linux glibc: definitions of the
//! `__isoc23_*` functions that AWS-LC's C code references when built
//! against glibc 2.38+ headers, so the library still loads on older glibc
//! (the OpenSearch image's 2.34). See that file's comment.

fn main() {
    println!("cargo:rerun-if-changed=native/glibc_compat.c");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if os != "linux" || env != "gnu" {
        return;
    }
    cc::Build::new()
        .file("native/glibc_compat.c")
        .std("c11")
        .link_lib_modifier("+whole-archive")
        .compile("lucene_ffi_glibc_compat");
}
