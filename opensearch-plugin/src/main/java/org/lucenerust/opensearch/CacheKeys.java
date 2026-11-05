/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.IndexReader;

import java.lang.reflect.Constructor;

/** Fresh {@link IndexReader.CacheKey}s for readers this plugin builds; Lucene keeps the constructor to itself. */
final class CacheKeys {
    private static final Constructor<IndexReader.CacheKey> CONSTRUCTOR;

    static {
        try {
            CONSTRUCTOR = IndexReader.CacheKey.class.getDeclaredConstructor();
            CONSTRUCTOR.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private CacheKeys() {}

    static IndexReader.CacheKey newKey() {
        try {
            return CONSTRUCTOR.newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("cannot create a reader cache key", e);
        }
    }
}
