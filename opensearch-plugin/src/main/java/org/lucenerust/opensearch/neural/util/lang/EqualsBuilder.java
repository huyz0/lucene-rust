/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.neural.util.lang;

import java.util.Objects;

/** commons-lang3's {@code EqualsBuilder}: field-by-field equality, arrays compared by content. */
public final class EqualsBuilder {
    private boolean isEquals = true;

    public EqualsBuilder append(Object lhs, Object rhs) {
        if (isEquals) {
            isEquals = Objects.deepEquals(lhs, rhs);
        }
        return this;
    }

    public boolean isEquals() {
        return isEquals;
    }
}
