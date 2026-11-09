/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.util;

import org.opensearch.Version;

/** The stats API's cluster version gates, from neural-search's {@code MinClusterVersionUtil}. */
public final class MinClusterVersionUtil {
    public static final Version MINIMAL_SUPPORTED_VERSION_STATS_CATEGORY_FILTERING = Version.V_3_1_0;
    public static final Version MINIMAL_SUPPORTED_VERSION_METRICS_STATS = Version.V_3_3_0;

    private MinClusterVersionUtil() {}

    public static boolean isClusterOnOrAfterMinReqVersionForStatCategoryFiltering() {
        return NeuralSearchClusterUtil.instance().getClusterMinVersion().onOrAfter(MINIMAL_SUPPORTED_VERSION_STATS_CATEGORY_FILTERING);
    }

    public static boolean isClusterOnOrAfterMinReqVersionForMetricStats() {
        return NeuralSearchClusterUtil.instance().getClusterMinVersion().onOrAfter(MINIMAL_SUPPORTED_VERSION_METRICS_STATS);
    }
}
