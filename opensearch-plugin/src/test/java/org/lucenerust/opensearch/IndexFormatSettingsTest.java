/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.lucenerust.opensearch.engine.IndexFormatSettings;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.apache.lucene.util.Version;
import org.junit.Test;
import org.opensearch.common.settings.Settings;

public class IndexFormatSettingsTest {
    private static final Version L105 = Version.fromBits(10, 5, 1);
    private static final Version L106 = Version.fromBits(10, 6, 0);

    @Test
    public void autoFollowsTheHostLucene() {
        assertEquals(105, IndexFormatSettings.resolve("auto", "auto", L105));
        assertEquals(106, IndexFormatSettings.resolve("auto", "auto", L106));
        assertEquals(105, IndexFormatSettings.resolve("10.5", "auto", L106));
    }

    @Test
    public void indexValueBeatsNodeValueBeatsHost() {
        assertEquals(105, IndexFormatSettings.resolve("10.5", "10.6", L105));
        assertEquals(105, IndexFormatSettings.resolve("auto", "10.5", L105));
    }

    @Test
    public void aFormatNewerThanTheHostIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> IndexFormatSettings.resolve("10.6", "auto", L105));
        assertThrows(IllegalArgumentException.class, () -> IndexFormatSettings.resolve("auto", "10.6", L105));
    }

    @Test
    public void theSettingRejectsOtherValues() {
        assertEquals("10.5", IndexFormatSettings.INDEX_FORMAT.get(Settings.builder().put("index.codec.index_format", "10.5").build()));
        assertThrows(
            IllegalArgumentException.class,
            () -> IndexFormatSettings.INDEX_FORMAT.get(Settings.builder().put("index.codec.index_format", "11").build())
        );
    }
}
