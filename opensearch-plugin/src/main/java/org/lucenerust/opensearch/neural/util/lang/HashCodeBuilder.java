/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.neural.util.lang;

import java.util.Arrays;

/** commons-lang3's {@code HashCodeBuilder}: a hash over appended fields, arrays by content. */
public final class HashCodeBuilder {
    private int total = 17;

    public HashCodeBuilder append(Object value) {
        int h = value == null ? 0 : value.getClass().isArray() ? Arrays.deepHashCode(new Object[] { value }) : value.hashCode();
        total = total * 37 + h;
        return this;
    }

    public int toHashCode() {
        return total;
    }
}
