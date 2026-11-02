/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.util.function.Predicate;

import org.apache.lucene.store.Directory;

/** Test access to {@link NativeStores}'s class-loader rule, with the rule replaced. */
public final class NativeStoresAccess {
    private NativeStoresAccess() {}

    public static String refusal(Directory dir, Predicate<Class<?>> trusted) {
        return NativeStores.refusal(dir, trusted);
    }
}
