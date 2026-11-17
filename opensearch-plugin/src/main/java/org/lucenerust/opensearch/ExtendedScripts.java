package org.lucenerust.opensearch;

import org.opensearch.common.settings.Setting;

/**
 * Whether the script shapes that run slower natively than on OpenSearch's own implementation are
 * still taken natively: Painless scripts over {@code date} values ({@code java.time} calls through
 * {@code def}), {@code random_score} seeded by {@code _id}, and Lucene expressions that score,
 * sort or filter a whole index. Off, those requests run on OpenSearch's script and function score
 * classes; the other script and function score shapes (and expression aggregations) stay native
 * either way.
 */
public final class ExtendedScripts {
    public static final Setting<Boolean> ENABLED = Setting.boolSetting(
        "index.script.accelerated_extended",
        false,
        Setting.Property.IndexScope,
        Setting.Property.Dynamic
    );

    private ExtendedScripts() {}
}
