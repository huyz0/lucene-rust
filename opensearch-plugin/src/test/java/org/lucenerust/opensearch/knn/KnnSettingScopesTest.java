/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Setting.Property;

/** The two k-NN index settings stock marks {@code Final} and {@code UnmodifiableOnRestore}: a snapshot restore may not change them. */
public class KnnSettingScopesTest {
    @Test
    public void indexKnnCannotChangeOnRestore() {
        assertFixed(KnnSettings.KNN_INDEX);
    }

    @Test
    public void derivedSourceCannotChangeOnRestore() {
        assertFixed(KnnSettings.DERIVED_SOURCE);
    }

    private static void assertFixed(Setting<?> setting) {
        assertTrue(setting.getKey(), setting.hasIndexScope());
        assertTrue(setting.getKey(), setting.getProperties().contains(Property.Final));
        assertTrue(setting.getKey(), setting.getProperties().contains(Property.UnmodifiableOnRestore));
    }
}
