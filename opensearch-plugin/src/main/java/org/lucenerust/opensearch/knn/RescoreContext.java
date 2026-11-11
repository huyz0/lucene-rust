/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import java.util.Objects;

/** k-NN's {@code RescoreContext}: a query's {@code rescore} (oversample factor, on or off). */
public final class RescoreContext {
    public static final float DEFAULT_OVERSAMPLE_FACTOR = 1.0f;
    public static final float MAX_OVERSAMPLE_FACTOR = 100.0f;
    public static final float MIN_OVERSAMPLE_FACTOR = 1.0f;

    private final float oversampleFactor;
    private final boolean rescoreEnabled;

    public RescoreContext(float oversampleFactor, boolean rescoreEnabled) {
        this.oversampleFactor = oversampleFactor;
        this.rescoreEnabled = rescoreEnabled;
    }

    public float getOversampleFactor() {
        return oversampleFactor;
    }

    public boolean isRescoreEnabled() {
        return rescoreEnabled;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof RescoreContext r && oversampleFactor == r.oversampleFactor && rescoreEnabled == r.rescoreEnabled;
    }

    @Override
    public int hashCode() {
        return Objects.hash(oversampleFactor, rescoreEnabled);
    }
}
