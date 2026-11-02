/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 *
 * glibc 2.38 headers redirect sscanf/strtol and friends to new __isoc23_*
 * symbols when C code is compiled with _GNU_SOURCE, as AWS-LC is (the
 * encrypted store's crypto, lucene-store-crypto). A library built on a newer
 * system would then need glibc 2.38, and refuse to load on the OpenSearch
 * image (glibc 2.34). These definitions satisfy those references inside
 * liblucene_ffi itself (hidden, never exported), calling the classic
 * functions: this file is compiled without _GNU_SOURCE, so its own calls are
 * not redirected. The only difference C23 makes is accepting a "0b" binary
 * prefix, which AWS-LC never parses.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define HIDDEN __attribute__((visibility("hidden")))

HIDDEN int __isoc23_vsscanf(const char *s, const char *fmt, va_list ap) {
    return vsscanf(s, fmt, ap);
}

HIDDEN int __isoc23_sscanf(const char *s, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsscanf(s, fmt, ap);
    va_end(ap);
    return n;
}

HIDDEN long __isoc23_strtol(const char *s, char **end, int base) {
    return strtol(s, end, base);
}

HIDDEN unsigned long __isoc23_strtoul(const char *s, char **end, int base) {
    return strtoul(s, end, base);
}

HIDDEN long long __isoc23_strtoll(const char *s, char **end, int base) {
    return strtoll(s, end, base);
}

HIDDEN unsigned long long __isoc23_strtoull(const char *s, char **end, int base) {
    return strtoull(s, end, base);
}
