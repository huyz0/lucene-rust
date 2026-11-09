/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.stats.common;

/**
 * Interface for the type of stat. Used for stat type metadata
 */
public interface StatType {

    /**
     * Get the name of the stat type containing info about the type and how to process it
     * @return name of the stat type
     */
    String getTypeString();
}
