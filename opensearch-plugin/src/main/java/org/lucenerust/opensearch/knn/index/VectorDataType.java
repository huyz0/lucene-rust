/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.index;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.lucenerust.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/** k-NN's {@code VectorDataType}: a knn_vector field's {@code data_type} and its parsing. */
public enum VectorDataType {
    BINARY("binary"),
    BYTE("byte"),
    FLOAT("float"),
    HALF_FLOAT("half_float");

    public static final String SUPPORTED_VECTOR_DATA_TYPES = Arrays.stream(values()).map(VectorDataType::getValue).collect(Collectors.joining(","));

    public static final VectorDataType DEFAULT = FLOAT;

    private final String value;

    VectorDataType(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public static VectorDataType get(String vectorDataType) {
        Objects.requireNonNull(
            vectorDataType,
            String.format(Locale.ROOT, "[%s] should not be null. Supported types are [%s]", VECTOR_DATA_TYPE_FIELD, SUPPORTED_VECTOR_DATA_TYPES)
        );
        try {
            return VectorDataType.valueOf(vectorDataType.toUpperCase(Locale.ROOT));
        } catch (Exception e) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Invalid value provided for [%s] field. Supported values are [%s]",
                    VECTOR_DATA_TYPE_FIELD,
                    SUPPORTED_VECTOR_DATA_TYPES
                )
            );
        }
    }
}
