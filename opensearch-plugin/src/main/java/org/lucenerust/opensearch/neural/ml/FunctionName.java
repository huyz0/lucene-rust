/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.ml;

import java.util.Locale;

/** ml-commons' {@code FunctionName}: a model's algorithm, and a remote model's {@code model_type}. */
public enum FunctionName {
    LINEAR_REGRESSION,
    KMEANS,
    AD_LIBSVM,
    SAMPLE_ALGO,
    LOCAL_SAMPLE_CALCULATOR,
    FIT_RCF,
    BATCH_RCF,
    ANOMALY_LOCALIZATION,
    RCF_SUMMARIZE,
    LOGISTIC_REGRESSION,
    TEXT_EMBEDDING,
    METRICS_CORRELATION,
    REMOTE,
    SPARSE_ENCODING,
    SPARSE_TOKENIZE,
    TEXT_SIMILARITY,
    QUESTION_ANSWERING,
    AGENT,
    CONNECTOR;

    public static FunctionName from(String value) {
        try {
            return FunctionName.valueOf(value.toUpperCase(Locale.ROOT));
        } catch (Exception e) {
            throw new IllegalArgumentException("Wrong function name");
        }
    }
}
