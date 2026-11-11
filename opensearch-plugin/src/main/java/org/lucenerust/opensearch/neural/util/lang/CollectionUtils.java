/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.neural.util.lang;

import java.util.Collection;
import java.util.Map;

/** The commons-collections {@code CollectionUtils} and {@code MapUtils} checks neural search calls. */
public final class CollectionUtils {
    private CollectionUtils() {}

    public static boolean isEmpty(Collection<?> c) {
        return c == null || c.isEmpty();
    }

    public static boolean isNotEmpty(Collection<?> c) {
        return isEmpty(c) == false;
    }

    public static boolean isEmpty(Map<?, ?> m) {
        return m == null || m.isEmpty();
    }
}
