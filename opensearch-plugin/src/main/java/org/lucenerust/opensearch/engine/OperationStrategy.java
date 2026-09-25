/*
 * Derived from OpenSearch 3.8.0 by opensearch-plugin/tools/derive_engine.py -- do not edit;
 * change the script and re-run it. The original notice follows.
 */
/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.lucenerust.opensearch.engine;

import org.opensearch.index.engine.*;

import java.util.Objects;
import java.util.Optional;

/**
 * Encapsulates the execution strategy for an engine operation (e.g. indexing or deletion).
 *
 * @opensearch.internal
 */
@SuppressWarnings({ "deprecation", "try", "unchecked", "rawtypes", "cast", "static", "fallthrough", "this-escape", "removal" }) // derived code, linted upstream
public class OperationStrategy {

    public final boolean executeOpOnEngine;
    public final boolean addStaleOpToEngine;
    public final long version;
    public final Optional<Engine.Result> earlyResultOnPreFlightError;
    public final int reservedDocs;

    public OperationStrategy(
        boolean executeOpOnEngine,
        boolean addStaleOpToEngine,
        long version,
        Engine.Result earlyResultOnPreFlightError,
        int reservedDocs
    ) {
        this.executeOpOnEngine = executeOpOnEngine;
        this.addStaleOpToEngine = addStaleOpToEngine;
        this.version = version;
        this.reservedDocs = reservedDocs;
        this.earlyResultOnPreFlightError = earlyResultOnPreFlightError == null
            ? Optional.empty()
            : Optional.of(earlyResultOnPreFlightError);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OperationStrategy that = (OperationStrategy) o;
        return executeOpOnEngine == that.executeOpOnEngine
            && addStaleOpToEngine == that.addStaleOpToEngine
            && version == that.version
            && reservedDocs == that.reservedDocs;
    }

    @Override
    public int hashCode() {
        return Objects.hash(executeOpOnEngine, addStaleOpToEngine, version, reservedDocs);
    }
}
