/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

/**
 * The native aggregation values hold one the Java aggregator refuses with an error of its own (a NaN
 * for a t-digest, a negative value for an HDR histogram): the search re-runs on Lucene, which raises
 * that error, rather than counting a native failure.
 */
final class JavaRefuses extends RuntimeException {
    JavaRefuses(String message, Throwable cause) {
        super(message, cause);
    }
}
