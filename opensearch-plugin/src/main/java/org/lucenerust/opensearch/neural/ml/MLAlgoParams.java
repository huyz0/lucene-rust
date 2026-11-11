/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml;

import java.util.Map;

/** ml-commons' {@code MLAlgoParams}: the {@code parameters} object of a predict request. */
public interface MLAlgoParams {
    /** The predict body's {@code parameters}, as ml-commons' REST predict takes them. */
    Map<String, Object> toParameters();
}
