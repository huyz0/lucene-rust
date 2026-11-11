/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.common;

/** The k-NN {@code KNNConstants} its MMR and source-excludes processors read. */
public final class KNNConstants {
    private KNNConstants() {}

    public static final String KNN_METHOD = "method";
    public static final String TYPE = "type";
    public static final String METHOD_PARAMETER_SPACE_TYPE = "space_type";
    public static final String TOP_LEVEL_PARAMETER_SPACE_TYPE = METHOD_PARAMETER_SPACE_TYPE;
    public static final String MODEL_ID = "model_id";
    public static final String VECTOR_DATA_TYPE_FIELD = "data_type";
    public static final String MMR = "mmr";
    public static final String MMR_EXPLAIN = "mmr_explain";
    public static final String EXPLAIN = "explain";
    public static final String DIVERSITY = "diversity";
    public static final String CANDIDATES = "candidates";
    public static final String VECTOR_FIELD_PATH = "vector_field_path";
    public static final String VECTOR_FIELD_DATA_TYPE = "vector_field_data_type";
    public static final String VECTOR_FIELD_SPACE_TYPE = "vector_field_space_type";
    public static final String MMR_RERANK_CONTEXT = "mmr.rerank_context";
}
