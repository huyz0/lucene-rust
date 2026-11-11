/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.util.lang;

/** The part of commons-lang3's {@code Pair} neural-search uses: an immutable pair with value equality. */
public record Pair<L, R>(L left, R right) {
    public static <L, R> Pair<L, R> of(L left, R right) {
        return new Pair<>(left, right);
    }

    public L getLeft() {
        return left;
    }

    public R getRight() {
        return right;
    }
}
